"""Protected Task5 assembly tests; fake science only, real opaque RunStore."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cipher_lab_plugins.speck_study.config import SpeckStudyRecipe
from cipher_lab_plugins.speck_study.tests.speck_method_fakes import (
    Meter,
    Private,
    Resources,
    Sampler,
    Teacher,
    np,
    science_guard,  # noqa: F401
)
from cipher_lab_plugins.speck_study.tests.speck_study_fakes import torch


def test_numpy_tensor_adapter_per_row_keys_not_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    from cipher_lab_plugins.speck_study.application import CipherAdapter
    seen = []
    def fake(points: Any, keys: Any, rounds: int) -> Any:
        seen.append((points.dtype, keys.shape, rounds, keys.tolist()))
        return points ^ 1
    import cipher_lab.ciphers.speck_tensor as cipher
    monkeypatch.setattr(cipher, 'encrypt_tensor', fake)
    adapter = CipherAdapter(torch, 'cpu')
    keys = np.array([[1,2,3,4],[5,6,7,8]], dtype=np.uint16)
    out = adapter(np.array([0,0xffffffff],dtype=np.uint32), keys, 8)
    assert out.dtype == np.uint32 and out.tolist() == [1,0xfffffffe]
    assert seen == [(torch.int64,(2,4),8,[[1,2,3,4],[5,6,7,8]])]
    assert set(vars(adapter)) == {'torch','device'}


def test_formal_r0_public_literal_only_counted_not_development_cipher() -> None:
    from cipher_lab_plugins.speck_study.application import formal_r0
    class Data:
        def __init__(self) -> None:
            self.calls: list[tuple[str,int,int]] = []
        def public_r0(self, points: Any, keys: Any, rounds: int, *, implementation: str) -> Any:
            assert keys.dtype == np.uint16
            self.calls.append((implementation, len(points), rounds))
            if rounds == 22:
                assert points.tolist() == [0x6574694c]
                assert keys.tolist() == [0x1918,0x1110,0x0908,0x0100]
                return np.array([0xa86842f2], dtype=np.uint32)
            return points.copy()  # fake, no full encryption
    d = Data()
    assert formal_r0(d)
    assert d.calls == [('reference',1,22),('batch',1,22),
                       ('reference',3,8),('batch',3,8)]
    assert sum(n for _,n,_ in d.calls) == 8


def test_formal_r0_mismatch_fail_closed_no_fallback() -> None:
    from cipher_lab_plugins.speck_study.application import formal_r0
    class Data:
        def public_r0(self, *args: Any, **kwargs: Any) -> Any:
            return np.array([0],dtype=np.uint32)
    assert formal_r0(Data()) is False


def test_prepared_assembly_real_data_fake_r_mismatch_closes_teacher_without_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uuid import UUID

    import cipher_lab.artifacts as artifacts
    from cipher_lab.artifacts import RunStore, read_saved
    from cipher_lab_plugins.speck_study.application import assemble
    from cipher_lab_plugins.speck_study.data import StudyData
    from cipher_lab_plugins.speck_study.storage import StudyStore
    monkeypatch.setattr(artifacts, 'uuid4', lambda: UUID(int=1))
    r = SpeckStudyRecipe.synthetic_recipe(r_bits=3,b_bits=3,initial_depth=1,
        pairs_per_layer=8,r_classification_size=8,r_coef_size=8,r_agreement_size=8,
        b_coef_size=8,b_dev_size=8,b_test_size=8)
    resources = Resources()
    t = Teacher(resources)
    store = RunStore.claim(tmp_path,'fake-assembly',output_bytes=10_000_000,needs_models=True)
    store.write_snapshot({'summary':{'backend_identity':'speck-study'}})
    from cipher_lab_plugins.speck_study.tests.speck_method_fakes import Cipher
    data, invoke = assemble(r, StudyStore(store,tmp_path,'fake-assembly'),
        Cipher(), Private(), Sampler(), Meter(resources), resources, t)
    assert isinstance(data,StudyData)
    invoke()  # fake classification fails fixed R gate; no full cipher/NN/random
    saved = read_saved(tmp_path,'fake-assembly')
    assert saved['status'] == 'SUCCEEDED'
    assert isinstance(saved['summary'],dict)
    assert saved['summary']['reason'] == 'REPRODUCTION_MISMATCH'
    assert saved['summary']['scientific_validation'] is False
    assert resources.acked and t.closed and data.closed


def test_closing_real_protocol_missing_request_not_forged_ack() -> None:
    from cipher_lab_plugins.speck_study.closing import ClosingChannel
    class Api:
        def __init__(self) -> None:
            self.states = {1:False,2:False}
        def set(self, handle: int) -> None:
            self.states[handle] = True
        def poll(self, handle: int) -> bool:
            return self.states[handle]
    api = Api()
    channel = ClosingChannel(api,1,2)
    assert not channel.requested() and not channel.acknowledged()
    with pytest.raises(RuntimeError):
        channel.acknowledge(saved=True,quiesced=False)
    channel.request()
    assert channel.requested() and not channel.acknowledged()
    channel.acknowledge(saved=True,quiesced=True)
    assert channel.acknowledged()


def test_default_launch_explicit_environment_blocked_before_science(tmp_path: Path) -> None:
    from cipher_lab_plugins.speck_study.application import require_native_launch
    with pytest.raises(RuntimeError,match='ENVIRONMENT_BLOCKED'):
        require_native_launch()


@pytest.mark.parametrize('result',[128,0xffffffff])
def test_windows_event_fault_not_ack_or_timeout(result: int) -> None:
    from cipher_lab_plugins.speck_study.closing import WindowsEvents
    class Api:
        def WaitForSingleObject(self, handle: int, timeout: int) -> int:
            assert timeout == 0
            return result
        def SetEvent(self, handle: int) -> int:
            return 0
    events = WindowsEvents(Api())
    with pytest.raises(RuntimeError,match='TRANSPORT_FAILED'):
        events.poll(1)
    with pytest.raises(RuntimeError,match='TRANSPORT_FAILED'):
        events.set(1)
