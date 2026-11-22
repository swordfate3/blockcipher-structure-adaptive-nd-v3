"""Four-task single-pair method under the innovation1 physical-effect guards."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CONFIG = Path('configs/innovation1-four-task-single-pair.json')


def test_check_exact_budget_and_pinned_skinny(capsys: pytest.CaptureFixture[str],
                                             tmp_path: Path) -> None:
    from cipher_lab_plugins.innovation1.cli import main
    from cipher_lab_plugins.innovation1.layout_probe.config import FIXED_BCE

    assert main(['check', '--config', str(CONFIG)]) == 0
    result = json.loads(capsys.readouterr().out)
    n, recipe = result['estimate'], result['recipe']
    assert n['optimizer_updates'] == 72_000
    assert n['forward_rows'] == 401_048_576
    assert n['fresh_pool_real_blocks'] == 60_000_000
    assert n['test_real_blocks'] == 2_097_152
    assert n['real_blocks'] == 62_097_152
    assert n['dev_tail_rows'] == 64
    assert n['fits'] == 1 and n['retries'] == 0
    assert n['artifact_peak_bound_bytes'] < 2**30
    assert recipe['data']['SKINNY']['source'] == FIXED_BCE.to_dict()['data']['source']
    assert recipe['data']['SKINNY']['shuffle_namespace'] == FIXED_BCE.shuffle_namespace
    assert [v['cipher'] for v in recipe['tasks']] == ['SIMECK', 'PRESENT', 'SIMON', 'SKINNY']
    assert recipe['resources']['wall_seconds'] == 43200
    assert recipe['resources']['cpu_seconds'] == 115200
    for section, field, changed in [('training', 'pairs_per_group', True),
                                    ('optimizer', 'schedule', 'cycle-cosine-then-constant'),
                                    ('data', 'SKINNY', {'protocol': 'fresh'})]:
        alternative = json.loads(CONFIG.read_text())
        alternative[section][field] = changed
        path = tmp_path / 'changed.json'
        path.write_text(json.dumps(alternative), encoding='ascii')
        assert main(['check', '--config', str(path)]) == 2


def test_all_tasks_use_low_word_first_one_key_per_pair() -> None:
    from collections import deque
    from dataclasses import replace

    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE
    from cipher_lab_plugins.innovation1.single_train.data import Provider

    tr = replace(FOUR_SINGLE.training, batches_per_epoch=9, batch_groups=4, dev_groups=4)
    key_bytes = deque([16, 10, 16, 8])
    expected = [(16, 0x40), (7, 0xD00000), (13, 0x40), (8, 0x2000)]
    for task, (rounds, difference) in zip(FOUR_SINGLE.tasks, expected, strict=True):
        width = key_bytes.popleft()
        calls: list[tuple[int, ...]] = []
        draws: list[int] = []
        debits: list[tuple[str, int]] = []

        def fake_key(n: int) -> bytes:
            assert n == width
            draws.append(n)
            assert len(draws) == len(calls) + 1
            return bytes(range(1, n + 1))

        def fake_oracle(points: tuple[int, ...], key: tuple[int, ...],
                        actual_rounds: int) -> tuple[int, ...]:
            assert actual_rounds == rounds and len(points) == 2
            raw = bytes(range(1, width + 1))
            assert key == tuple(int.from_bytes(raw[max(i-4, 0):i], 'big')
                                for i in range(width, 0, -4))
            calls.append(points)
            return 0x0123456789ABCDEF, 0xFEDCBA9876543210

        from types import SimpleNamespace
        meter = SimpleNamespace(check=lambda: None, reserve=lambda *v: debits.append(v),
                                complete=lambda *v: debits.append(v))
        provider = Provider(replace(FOUR_SINGLE, training=tr), fake_oracle, fake_key, meter,
                            streams=FOUR_SINGLE.streams, task=task)
        batch = provider.train(0)
        assert len(draws) == len(calls) == 4
        assert debits == [('real_blocks', 2)] * 8
        assert batch.ciphertext[0].tobytes().hex() == '0123456789abcdeffedcba9876543210'
        assert [(a ^ b) == difference for a, b in calls] == list(batch.labels[:, 0] == 1)


def test_fresh_pool_finishes_before_replay_and_keeps_stratified_split() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np
    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE
    from cipher_lab_plugins.innovation1.single_train.data import FixedPairPool, PairBatch
    from plugin_isolation import forbidden

    recipe = replace(FOUR_SINGLE, training=replace(FOUR_SINGLE.training, epochs=2,
                     batch_groups=4, batches_per_epoch=9, dev_groups=4))
    generated: list[int] = []

    class Pool(FixedPairPool):
        def _collect(self, split: str, index: int, size: int) -> PairBatch:
            assert (split, size) == ('train', 4)
            generated.append(index)
            x = np.repeat(np.arange(index*4, index*4+4, dtype=np.uint8), 16)
            return PairBatch(x.reshape(4, 1, 2, 8), np.array([[1], [0], [0], [1]], np.uint8),
                             split, index, recipe.namespace)

    progress: list[int] = []
    p = Pool(recipe, forbidden, forbidden, SimpleNamespace(check=lambda: None),
             streams=recipe.streams, task=recipe.tasks[0], shuffle_namespace=recipe.namespace,
             progress=progress.append)
    train = p.prepare_pool()
    assert generated == list(range(10)) and progress == list(range(1, 11))
    assert train.ciphertext[:, 0, 0, 0].tolist() == list(range(36))
    dev = p.panel('dev')
    assert dev.ciphertext[:, 0, 0, 0].tolist() == [37, 36, 38, 39]
    with pytest.raises(ValueError, match='POOL_NOT_READY'):
        p.train(0)
    p.activate()
    permutations = [[int(x) for cycle in range(e*9, (e+1)*9)
                     for x in p.train(cycle).ciphertext[:, 0, 0, 0]] for e in range(2)]
    assert all(sorted(perm) == list(range(36)) for perm in permutations)
    assert permutations[0] != permutations[1] and len(generated) == 10


@pytest.mark.parametrize('select_nibble', [False, True])
def test_card_selects_column_only_for_skinny_nibble_without_cross_pair_pooling(
        select_nibble: bool) -> None:
    from types import SimpleNamespace
    from typing import Any

    import torch
    from cipher_lab_plugins.innovation1.four_single.models import SinglePairM
    from cipher_lab_plugins.innovation1.layout_probe.layout import nibble_indices
    from plugin_isolation import forbidden

    from cipher_lab.representations.tasks import public_features

    class FakeNetwork:
        def __getitem__(self, index: int | slice) -> Any:
            return (lambda x: x[:, :1]) if index == 0 else (lambda x: x)

    class Stem(torch.nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind
            self.indices = nibble_indices('row')
            self.channels = 1
            # Fake the operators while running the adapter's actual gather math.
            self.network = FakeNetwork()
            self.relation_projection = lambda x: x[:, :1]

        def forward(self, raw: Any, channel: Any) -> Any:
            calls.append(self.kind)
            return raw[:, :, :1]

    def represent(x: Any, card: Any, *, rounds: int) -> Any:
        assert card is public and rounds == wanted_rounds
        return SimpleNamespace(canonical=x[:, :, [0, 1, 0]], channel=x[:, :, :1] * 0)

    def route(card: Any) -> Any:
        assert card is public
        return SimpleNamespace(indices=torch.tensor([2 if select_nibble else 1, 0]),
                               weights=torch.tensor([.25, .75]))

    calls: list[str] = []
    fake: Any = torch.nn.Module()
    fake.register_parameter('anchor', torch.nn.Parameter(torch.zeros(1)))
    fake.stems = torch.nn.ModuleDict({name: Stem(name) for name in ('local', 'word', 'nibble')})
    fake.spec = SimpleNamespace(channels=1)
    fake.representation, fake.router = represent, route
    fake.suffix = lambda x: x + 1
    fake.readout = SimpleNamespace(forward=forbidden, base=lambda x: x[:, 64-1:64],
                                   residual=lambda x: x[:, 63:64].square())
    model = SinglePairM(fake)
    ids = torch.arange(1., 9.)
    bits = ids[:, None, None, None].repeat(1, 1, 2, 64)
    column = nibble_indices('column')
    # Independently reviewed boundary: bit 63's predecessor nibble is 11.
    assert column[63].tolist() == [46, 47, 64, 62, 63, 64, 64, 64, 64]
    bits[:, :, :, 46] = 2 * ids[:, None, None]
    before = fake.stems['nibble'].indices.clone()
    for cipher, wanted_rounds in [('SIMECK', 16), ('PRESENT', 7), ('SIMON', 13), ('SKINNY', 8)]:
        calls.clear()
        public = torch.tensor(public_features(cipher))
        result = model(bits, public, rounds=wanted_rounds)
        special = cipher == 'SKINNY' and select_nibble
        assert calls == (['local'] if special else ['nibble' if select_nibble else 'word', 'local'])
        h = (1.25 * ids if special else ids) + 1
        torch.testing.assert_close(result, (h + h.square())[:, None])
        assert torch.equal(before, fake.stems['nibble'].indices)
    public = public.clone()
    public[-1] = .99
    with pytest.raises(ValueError, match='unreviewed public card'):
        model(bits, public, rounds=8)


def test_checkpoint_rejects_changed_public_tables_and_bound_identity() -> None:
    from copy import deepcopy
    from dataclasses import replace

    import torch
    from cipher_lab_plugins.innovation1.checkpoint import encode
    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE
    from cipher_lab_plugins.innovation1.four_single.models import build, geometry, load_state

    recipe = replace(FOUR_SINGLE, run_id='synthetic-four')
    model = build(recipe)
    assert sum(p.numel() for p in model.parameters()) == 1_087_509
    tables = geometry(model)
    assert set(tables) == {'skinny_column', 'shared.stems.nibble.indices',
                           'shared.stems.word.indices', 'shared.stems.local.indices'}
    identity: dict[str, object] = dict(method=recipe.method, run_id=recipe.run_id,
                    source={'fake/model.py': 'a'*64},
                    pools={'SKINNY': 'b'*64}, geometry=tables)
    raw = encode(identity, model.state_dict())
    load_state(model, raw, identity)
    for key in ('method', 'run_id', 'source', 'pools', 'geometry'):
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            load_state(model, raw, {**identity, key: 'wrong'})
    for name in tables:
        state = deepcopy(model.state_dict())
        state[name][0, 0] = (state[name][0, 0] + 1) % 65
        with pytest.raises(ValueError, match='CHECKPOINT_INVALID'):
            load_state(model, encode(identity, state), identity)
    assert model.skinny_column.device == torch.device('cpu')


@pytest.mark.parametrize('fault', ['', 'missing', 'manifest', 'disk', 'identity', 'save'])
def test_four_task_fit_barriers_and_saved_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  fault: str) -> None:
    import io
    from dataclasses import replace
    from hashlib import sha256
    from typing import Any, cast

    import numpy as np
    import torch
    from cipher_lab_plugins.innovation1 import application, cli
    from cipher_lab_plugins.innovation1.config import Task
    from cipher_lab_plugins.innovation1.four_single import training
    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE, make_policy
    from cipher_lab_plugins.innovation1.four_single.data import Pools
    from cipher_lab_plugins.innovation1.resources import Resources
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch
    from cipher_lab_plugins.innovation1.tests.test_innovation1_fixed_bce import _fixture

    from cipher_lab.artifacts import RunStore, read_saved
    from cipher_lab.execution import Clocks, Meter

    parent, prior, old_files = _fixture(tmp_path, fault if fault in ('missing', 'manifest') else '')
    replay = [prior.train(c).digest for c in range(27)]
    recipe = replace(FOUR_SINGLE, run_id='tiny-four', skinny_source=parent.pool_source,
                     training=replace(parent.training, epochs=4))
    root = tmp_path / 'current'
    clocks = Clocks()
    res = Resources(Meter(make_policy(recipe, clocks), clocks, lambda: 0))
    source = {'fake/code.py': 'a' * 64}
    store = RunStore.claim(root, 'tiny-four', output_bytes=2**30, needs_models=True,
                           model_slots=('M-best',), source_supplier=lambda: source)
    record = root / 'data/tiny-four/run.json'
    tasks = [t.cipher for t in recipe.tasks]
    stage = [0, False, False]  # actual updates, released, restored
    private_sizes: list[int] = []
    checkpoint_epochs: list[int] = []
    model_path = root / 'models/tiny-four/M-best.pt'

    def fake_private(size: int) -> bytes:
        assert fault not in ('missing', 'manifest')
        if stage[0]:
            assert stage[2]
        private_sizes.append(size)
        return bytes(size)

    def fake_oracle(blocks: tuple[int, ...], key: tuple[int, ...], r: int) -> tuple[int, ...]:
        return (r << 56 | blocks[0] & ((1 << 56) - 1), blocks[1])

    class Runtime:
        def metadata(self) -> dict[str, object]:
            return {'M-best': {'parameters': 1087509, 'geometry': {'column': 'c' * 64}}}

        def prepare(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
            if batch.split == 'train' and batch.namespace == prior.recipe.namespace:
                if batch.index < len(replay):
                    assert batch.digest == replay[batch.index]
            x, y = batch.tensors()
            # Explicitly synthetic scores with an identity for label/task auditing.
            x[:, 0, 0, 0] = y[:, 0]
            return x, y

        def update(self, slot: str, task: Task, x: torch.Tensor, y: torch.Tensor, lr: float,
                   resources: Resources) -> float:
            assert slot == 'M-best' and task.cipher == tasks[int(stage[0]) % 4]
            if not stage[0]:
                saved = json.loads(record.read_bytes())
                assert saved['summary']['pool_status'] == 'READY'
                assert list(sorted(saved['artifacts'])) == [f'dataset/{s}-{t}.npz'
                    for s in ('dev', 'train') for t in sorted(tasks[:-1])]
                assert len(private_sizes) == 120
                assert not model_path.exists()
                assert lr == .001
            if stage[0] == 143:
                assert lr == .0001
            stage[0] += 1
            resources.reserve('forward_rows', int(y.shape[0]))
            resources.complete('forward_rows', int(y.shape[0]))
            assert x.shape == (4, 1, 2, 64)
            return .7

        def predict(self, slot: str, task: Task, x: torch.Tensor) -> Any:
            assert slot == 'M-best' and len(x) == 4
            e = int(stage[0]) // 36
            weights = {1: [3, 0, 0, 0], 2: [1, 1, 1, 1], 3: [1, 1, 1, 1],
                       4: [2, 2, -1, 2]}
            a = 2 if stage[2] else weights[e][tasks.index(task.cipher)]
            return ((2*x[:, 0, 0, 0] - 1) * a)[:, None].numpy()

        def routes(self) -> dict[str, dict[str, float]]:
            return {t: {'local': .75, 'nibble': .25} for t in tasks}

        def checkpoint(self, slot: str, metadata: dict[str, Any]) -> bytes:
            assert slot == 'M-best'
            checkpoint_epochs.append(metadata['epoch'])
            assert metadata['pools']['SKINNY']['pool_source']['run_id'] == prior.recipe.run_id
            assert metadata['source'] == source and metadata['run_id'] == 'tiny-four'
            return json.dumps(metadata, sort_keys=True).encode()

        def release(self) -> None:
            assert int(stage[0]) == 144
            stage[1] = True
            if fault == 'disk':
                model_path.write_bytes(b'x')

        def restore(self, slot: str, raw: bytes, metadata: dict[str, Any]) -> None:
            assert stage[1] and slot == 'M-best' and metadata['epoch'] == 2
            assert raw == model_path.read_bytes() and json.loads(raw) == metadata
            if fault == 'identity':
                raise ValueError('CHECKPOINT_INVALID')
            stage[2] = True

    original = RunStore._atomic

    def broken_save(self: RunStore, target: Path, content: bytes) -> None:
        if target.name == 'dev-SIMON.npz':
            raise OSError('injected storage failure')
        original(self, target, content)

    if fault == 'save':
        monkeypatch.setattr(RunStore, '_atomic', broken_save)
    # The real product dispatch wrapper must write all pool/ledger/terminal evidence.
    monkeypatch.setattr(training, 'assemble', lambda rec, resources: (
        Pools(rec, dict.fromkeys(tasks, fake_oracle), fake_private, resources),
        Runtime()))
    monkeypatch.setattr(training, 'official', lambda rec, run_id: rec)
    monkeypatch.setattr(application, 'official_four_single', lambda rec, run_id: rec)
    monkeypatch.setattr(application, 'prepare_device', lambda *_: None)
    application.execute(recipe, store, res, gpu_uuid='GPU-00000000-0000-0000-0000-000000000000',
                        run_id='tiny-four')
    for path, raw in old_files.items():
        assert path.read_bytes() == raw
    saved = json.loads(record.read_bytes())
    summary = saved['summary']
    if fault:
        assert saved['status'] == 'FAILED'
        assert not any(key.startswith('dataset/test-') for key in saved['artifacts'])
        assert not stage[2]
        if fault in ('missing', 'manifest', 'save'):
            assert not stage[0]
        if fault in ('missing', 'manifest'):
            assert not private_sizes and summary['reason'] == 'POOL_SOURCE_INVALID'
        return
    assert saved['status'] == 'SUCCEEDED'
    assert checkpoint_epochs == [1, 2]
    assert list(model_path.parent.iterdir()) == [model_path]
    assert len(summary['curves']) == 4 and summary['best']['epoch'] == 2
    assert summary['best']['routes'] == summary['curves'][1]['routes']
    assert summary['persisted_reload']['sha256'] == sha256(model_path.read_bytes()).hexdigest()
    assert private_sizes == [16]*40 + [10]*40 + [16]*40 + [16]*4 + [10]*4 + [16]*4 + [8]*4
    for kind in ('attempted', 'completed'):
        assert saved['counts'][f'{kind}_real_blocks'] == 272
        assert saved['counts'][f'{kind}_optimizer_updates'] == 144
        assert saved['counts'][f'{kind}_forward_rows'] == 656
    rows = [row for ref in summary['batch_refs']
            for row in json.loads((record.parent / ref).read_bytes())]
    assert len(rows) == 144 and all(row['completed'] for row in rows)
    assert [(row['cycle'], row['task']) for row in rows] == [
        (c, t) for c in range(36) for t in tasks]
    for cipher in tasks:
        ref = f'dataset/test-{cipher}.npz'
        raw = (record.parent / ref).read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            assert set(arrays.files) == {'metadata', 'ciphertext', 'labels', 'logits'}
            assert arrays['logits'][:, 0].tolist() == [
                2. if y else -2. for y in arrays['labels'][:, 0]]
            metadata = json.loads(arrays['metadata'].tobytes())
            assert metadata['best'] == summary['persisted_reload']
        assert summary['test'][cipher]['scores']['accuracy'] == 1.
        assert saved['artifacts'][ref]['sha256'] == sha256(raw).hexdigest()
    assert cast(dict[str, Any], read_saved(root, 'tiny-four'))['status'] == 'SUCCEEDED'
    assert cli._finish(store, root, 'tiny-four', resources=res) == 0
    assert (root / 'result/tiny-four/report.md').is_file()


def test_cli_refuses_pinned_pool_overlap_before_claim(capsys: pytest.CaptureFixture[str]) -> None:
    from cipher_lab_plugins.innovation1.cli import main
    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE

    pin = FOUR_SINGLE.skinny_source
    assert pin is not None
    gpu = 'GPU-00000000-0000-0000-0000-000000000000'
    assert main(['run', '--config', str(CONFIG), '--id', 'overlap', '--gpu-uuid', gpu,
                 '--root', pin.root + r'\must-not-claim']) == 2
    assert 'invalid settings' in capsys.readouterr().err


def test_live_runtime_constructs_only_the_one_seeded_m_without_running_layers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    from cipher_lab_plugins.innovation1.config import Recipe
    from cipher_lab_plugins.innovation1.four_single import runtime
    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE
    from cipher_lab_plugins.innovation1.models import TorchRuntime

    constructed: list[str] = []
    bind = TorchRuntime.__init__

    def fake_device(self: TorchRuntime, recipe: Recipe, device: str,
                    *, inference_only: bool = False) -> None:
        bind(self, recipe, device, inference_only=inference_only)  # no actual cuda or Adam
        self.device = torch.device('cpu')

    def no_adam(self: TorchRuntime, slot: str) -> None:
        assert list(self.models) == [slot]
        constructed.append(slot)

    monkeypatch.setattr(TorchRuntime, '__init__', fake_device)
    monkeypatch.setattr(TorchRuntime, 'add_adam', no_adam)
    before = torch.random.get_rng_state()
    model = runtime.Runtime(FOUR_SINGLE, lambda: None)
    assert torch.equal(before, torch.random.get_rng_state())
    assert constructed == ['M-best']
    assert model.metadata()['M-best']['parameters'] == 1_087_509
    assert {t.cipher: tuple(card.tolist()) for t, card in model.public.items()} == {
        'SIMECK': (0, 1, 0, .5, 1., 1., .125, 0, 0, 5/32, 1/32, 0, 0),
        'PRESENT': (0, 0, 1, .5, .625, .125, 1, .5, 0, 0, 0, 1, 0),
        'SIMON': (0, 1, 0, .5, 1., 1., .125, 0, 1/32, 8/32, 2/32, 0, 0),
        'SKINNY': (0, 0, 1, .5, .5, .125, 1, .5, 0, 0, 0, 0, 1),
    }


def test_eval_scores_512_rows_and_real_64_row_tail_once() -> None:
    from types import SimpleNamespace
    from typing import Any

    import numpy as np
    from cipher_lab_plugins.innovation1.four_single.config import FOUR_SINGLE
    from cipher_lab_plugins.innovation1.four_single.training import score
    from cipher_lab_plugins.innovation1.single_train.data import PairBatch

    labels = np.tile(np.array([[0], [1]], np.uint8), (288, 1))
    x = np.zeros((576, 1, 2, 8), np.uint8)
    x[:, 0, 0, 0] = np.where(labels[:, 0] == 1, 128, 0)
    panel = PairBatch(x, labels, 'dev', 0, FOUR_SINGLE.namespace)
    charged: list[tuple[str, int]] = []
    observed: list[int] = []
    positions: list[int] = []
    res: Any = SimpleNamespace(check=lambda: None, reserve=lambda *v: charged.append(v),
                          complete=lambda *v: charged.append(v))
    task = FOUR_SINGLE.tasks[1]

    def predict_one(slot: str, actual_task: object, bits: Any) -> Any:
        assert slot == 'M-best' and actual_task == task
        observed.append(len(bits))
        return (2 * bits[:, 0, 0, :1] - 1).numpy()

    runtime: Any = SimpleNamespace(prepare=PairBatch.tensors, predict=predict_one)
    scores, logits = score(task, panel, runtime, 512, res, positions.append)
    assert positions == [0, 512] and observed == [512, 64]
    assert charged == [('forward_rows', size) for size in (512, 512, 64, 64)]
    assert scores == {'groups': 576, 'accuracy': 1., 'bce': pytest.approx(.3132616875)}
    np.testing.assert_array_equal(logits, 2 * labels.astype(np.float64) - 1)
