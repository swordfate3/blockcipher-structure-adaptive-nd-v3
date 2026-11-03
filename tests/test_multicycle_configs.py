"""Task6 fixed multicycle config matrix, budget arithmetic and stage branch fixtures.

Pure config/JSON/arithmetic tests only: no model construction, no optimizer, no cipher,
no labels, no fitting. Stage decision helpers are test-local fixtures for documentation
of expected manual transitions; they are not a production scheduler and grant no authority.
"""
from pathlib import Path

from cipher_experiments.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1] / "configs"
N1_DIR = ROOT / "innovation2-multicycle-n1"
N2_DIR = ROOT / "innovation2-multicycle-n2"
SMOKE_DIR = ROOT / "innovation2-multicycle-smoke"

N1_NAMES = ("r-real", "r-null", "z-real", "z-null", "m-real", "m-null")
N2_METHODS = ("d", "f", "r", "fr", "s")
SMOKE_NAMES = (
    "n1-r", "n1-z", "n1-m", "n2-d", "n2-f", "n2-r", "n2-fr", "n2-s",
    "n3-r8-d", "n3-r8-r", "n3-r8-fr", "n3-r10-d", "n3-r10-r", "n3-r10-fr",
)

# Spec-documented parameter counts (brief 4.2); asserted as frozen constants, never
# recomputed by building models in these tests.
N2_PARAM_COUNTS = {"d": 24961, "f": 29057, "r": 24961, "fr": 29057, "s": 97}

MAX_FITS = 36
UPDATES_PER_FIT = 8192
PRESENTATIONS_PER_FIT = 2097152
ONLINE_DEV_PER_FIT = 32
TERMINAL_DEV_PER_FIT = 1
LABELS_PER_FIT_CAP = 139264
FROZEN_FORMAL_FITS = 16  # 6 N1 + 10 N2; N3/N4 stay conditional until candidate known
PAIRED_FAMILY_ALPHA = 0.05 / 18


def _common_formal(config: ExperimentConfig, *, data_seed: int, null_seed: int) -> None:
    assert config.dataset == "simon" and config.experiment == "prediction"
    assert (config.train_rows, config.dev_rows, config.test_rows) == (65536, 8192, 65536)
    assert config.train_pool_rows == 65536
    assert config.batch_size == 256 and config.epochs == 32 and config.dev_every_epochs == 1
    assert config.learning_rate == 0.001 and config.max_seconds == 1800
    assert config.device == "cuda:0"
    assert config.hidden_width == 128
    assert config.model_features == "full" and config.prediction_mode == "direct"
    assert config.target_bit == 0
    assert config.record_predictions is True
    assert (config.data_seed, config.null_seed, config.init_seed, config.key_seed) == (
        data_seed, null_seed, 17, 190502)


def test_n1_formal_fixed_matrix() -> None:
    paths = sorted(N1_DIR.glob("*.json"))
    assert {p.stem for p in paths} == set(N1_NAMES)
    assert len(paths) == 6
    for path in paths:
        config = load_config(path)
        _common_formal(config, data_seed=196101, null_seed=196102)
        assert config.rounds == 3
        assert config.plaintext_space == "high32-zero"
        assert config.plaintext_fixed_value == 2779096485  # X0 = 0xA5A5A5A5, B fixed
        assert config.label_expression == "xor2" and config.label_bits == (0, 8)
        expected_mode = "null" if path.stem.endswith("null") else "cipher"
        assert config.label_mode == expected_mode
        arm = path.stem.split("-")[0]
        if arm == "r":
            assert config.model_kind == "mlp" and config.background_init == "standard"
        else:
            assert config.model_kind == "public-mlp"
            assert config.public_input == "background-zero"
            assert config.public_readout == "direct"
            assert config.background_init == ("match-direct" if arm == "m" else "standard")
        assert len(config.name) <= 100


def test_n2_formal_fixed_matrix() -> None:
    paths = sorted(N2_DIR.glob("*.json"))
    assert {p.stem for p in paths} == {f"{m}-{side}"
                                       for m in N2_METHODS for side in ("real", "null")}
    assert len(paths) == 10
    for path in paths:
        config = load_config(path)
        _common_formal(config, data_seed=196201, null_seed=196202)
        assert config.rounds == 6
        assert config.plaintext_space == "full" and config.plaintext_fixed_value == 0
        assert config.label_expression == "bit" and config.label_bits == ()
        method, side = path.stem.split("-")
        assert config.label_mode == ("null" if side == "null" else "cipher")
        if method == "d":
            assert config.model_kind == "mlp"
        elif method == "f":
            assert config.model_kind == "public-mlp"
            assert config.public_input == "simon-interactions"
            assert config.public_readout == "direct"
            assert config.background_init == "standard"
        elif method == "r":
            assert config.model_kind == "public-mlp"
            assert config.public_input == "identity"
            assert config.public_readout == "zero-round-constant-residual"
        elif method == "fr":
            assert config.model_kind == "public-mlp"
            assert config.public_input == "simon-interactions"
            assert config.public_readout == "zero-round-constant-residual"
        else:
            assert method == "s"
            assert config.model_kind == "public-linear"
            assert config.public_input == "simon-interactions"
            assert config.public_readout == "zero-round-constant-residual"
            assert config.hidden_width == 128
        assert len(config.name) <= 100
    # Same-shape initial-weight pairing is documented as spec constants; no model build here.
    assert N2_PARAM_COUNTS == {"d": 24961, "f": 29057, "r": 24961, "fr": 29057, "s": 97}


def test_smoke_null_cpu_matrix() -> None:
    paths = sorted(SMOKE_DIR.glob("*.json"))
    assert {p.stem for p in paths} == set(SMOKE_NAMES)
    assert len(paths) == 14
    for path in paths:
        config = load_config(path)
        assert config.label_mode == "null" and config.device == "cpu"
        assert (config.train_rows, config.dev_rows, config.test_rows) == (8, 4, 4)
        assert config.train_pool_rows == 8
        assert config.batch_size == 4 and config.epochs == 1 and config.dev_every_epochs == 1
        assert config.hidden_width == 128 and config.max_seconds == 60
        assert config.learning_rate == 0.001 and config.model_features == "full"
        assert config.prediction_mode == "direct" and config.target_bit == 0
        assert config.init_seed == 17 and config.key_seed == 190502
        assert config.data_seed != 196101 and config.data_seed != 196201
        assert len(config.name) <= 100
        stem = path.stem
        if stem.startswith("n1-"):
            assert config.rounds == 3
            assert config.plaintext_space == "high32-zero"
            assert config.plaintext_fixed_value == 2779096485
            assert config.label_expression == "xor2" and config.label_bits == (0, 8)
            arm = stem.split("-")[1]
            if arm == "r":
                assert config.model_kind == "mlp"
            else:
                assert config.model_kind == "public-mlp"
                assert config.public_input == "background-zero"
                assert config.background_init == ("match-direct" if arm == "m" else "standard")
        else:
            assert config.label_expression == "bit" and config.label_bits == ()
            assert config.plaintext_space == "full"
            stage, tail = stem.split("-", 1)
            if stage == "n2":
                rounds, method = "r6", tail
            else:
                assert stage == "n3"
                rounds, method = tail.split("-", 1)
            assert config.rounds == int(rounds.removeprefix("r"))
            if method == "d":
                assert config.model_kind == "mlp"
            elif method == "r":
                assert config.model_kind == "public-mlp"
                assert config.public_input == "identity"
                assert config.public_readout == "zero-round-constant-residual"
            elif method == "f":
                assert config.model_kind == "public-mlp"
                assert config.public_input == "simon-interactions"
                assert config.public_readout == "direct"
            elif method == "s":
                assert config.model_kind == "public-linear"
                assert config.public_input == "simon-interactions"
                assert config.public_readout == "zero-round-constant-residual"
            else:
                assert method == "fr"
                assert config.model_kind == "public-mlp"
                assert config.public_input == "simon-interactions"
                assert config.public_readout == "zero-round-constant-residual"


def test_formal_budget_arithmetic_whole_cycle() -> None:
    # Whole formal matrix cap: at most 36 fits (not 36 architectures); not a development licence.
    assert FROZEN_FORMAL_FITS == 6 + 10
    conditional = 8 + 12  # N3 up to 8 fits, N4 up to 12 fits, only if branches open them
    assert FROZEN_FORMAL_FITS + conditional == MAX_FITS
    assert UPDATES_PER_FIT * MAX_FITS == 294912
    assert PRESENTATIONS_PER_FIT * MAX_FITS == 75497472
    online = ONLINE_DEV_PER_FIT * MAX_FITS
    terminal = TERMINAL_DEV_PER_FIT * MAX_FITS
    assert (online, terminal, online + terminal) == (1152, 36, 1188)
    assert LABELS_PER_FIT_CAP == 65536 + 8192 + 65536


def test_n4_three_blocks_fixed_alpha_no_reallocation() -> None:
    # Whole-cycle paired family: 3 blocks x 3 comparators x 2 metrics = 18 unique cells
    # (6 per block), not 18 per block. Each cell spends alpha=0.05/18 and the whole
    # family spends at most 0.05. Removing D self-comparisons is conservative: freed
    # cells stay unused, no alpha is reallocated to remaining comparisons.
    blocks = ("K1", "K2", "K3")
    comparators = ("train_constant", "q_reference_constant", "same_stage_d")
    metrics = ("acc", "bce")
    cells = {(block, comparator, metric)
             for block in blocks for comparator in comparators for metric in metrics}
    assert len(cells) == 18
    assert all(sum(1 for cell in cells if cell[0] == block) == 6 for block in blocks)
    cell_alpha = 0.05 / 18
    assert abs(sum(cell_alpha for _ in cells) - 0.05) < 1e-12
    # D is never self-compared; removing those cells must not raise other cells' alpha.
    assert "self" not in comparators
    assert all(abs(cell_alpha - PAIRED_FAMILY_ALPHA) < 1e-18 for _ in cells)


# Versioned execution/spec doc (must be inside Git, not an ignored validation/ artifact).
EXECUTION_DOC = Path(__file__).resolve().parents[1] / "docs/INNOVATION2-MULTICYCLE-EXECUTION.md"


def test_execution_doc_train_command_documents_required_output() -> None:
    # Actual CLI contract: train requires --output pointing at a non-existent run dir.
    text = EXECUTION_DOC.read_text(encoding="utf-8")
    assert "cipher-experiments train --config <config.json> --output runs/<new-run-id>" in text
    assert "输出目录必须不存在" in text
    # evaluate is only an entry point here; no fabricated full command is documented.
    assert "cipher-experiments evaluate --config" not in text
    assert "cipher-experiments evaluate" in text
    # check neither trains nor runs evaluate.
    assert "cipher-experiments check --config" in text and "不训练" in text
    # Statistics are frozen by design §7, not the earlier roadmap proposal.
    assert "全周期所有 ≤18 格合计 CPU **30 分钟** / wall **30 分钟**" in text
    assert "每格 CPU" not in text and "不是每格 30 分钟" in text
    assert ("superpowers/specs/2026-11-02-innovation2-multicycle-design.md"
            "#7-预测证据统计与确认比较族") in text
    assert "INNOVATION2-MULTICYCLE-ROADMAP-PROPOSAL.md" not in text
    for frozen in ("10000", "PCG64", "196801 + 100*block + 10*comparator + metric",
                   "alpha = .05/18", 'method="linear"', "64"):
        assert frozen in text
    assert "F／FR：29057；R：24961；S：97；D：24961" in text


def test_execution_doc_current_scope_approval_and_prohibitions() -> None:
    # Current explicit scope: the whole N1->N2->conditional N3/N4 science cycle is approved
    # within fixed caps; installation and the integral branch stay out of scope.
    text = EXECUTION_DOC.read_text(encoding="utf-8")
    assert "已批准" in text and "科学周期" in text
    assert "36" in text and "294912" in text and "75497472" in text
    assert "2506752" in text and "1800" in text
    assert "18 小时" in text and "30 分钟" in text
    assert "**不含安装**" in text and "**不含积分支线**" in text
    for permission in ("精确路径 Git 发布", "SSH 调用远端已有 CLI", "GPU fit",
                       "SSH/SCP 取回", "保存预测的配对统计"):
        assert permission in text
    assert "不 commit／push、不同步远端" not in text
    assert "不用 Git 回传结果" in text
    assert "不安装 solver／checker／新依赖" in text
    assert "不启动积分支线（I0–I3）或任何 solver／checker 调用" in text
    # Past Task6/Task7 engineering (14 null fits, 28 updates / 112 presentations) is recorded
    # as already-completed history, never re-run and never asserted as current authority.
    assert "28 updates / 112 presentations" in text
    assert "不重跑" in text
    # TEST is served by the normal CLI: no technical sealing may be claimed.
    assert "不封存" in text or "没有技术封存" in text


# ---- Stage decision fixtures: pure documentation of expected manual transitions. ----
# These helpers compute expected branch outcomes from already measured DEV quantities.
# They are not a controller: nothing here starts a run, checks permissions, or schedules work.


def capability_gate(train_majority_acc: float, train_bce: float, q_bce: float,
                    dev_acc: float, dev_bce: float) -> bool:
    """N2/N3 ability gate: beat TRAIN majority by >=.02 ACC and both TRAIN and q BCE by .005."""
    return (dev_acc >= train_majority_acc + 0.02
            and dev_bce <= train_bce - 0.005
            and dev_bce <= q_bce - 0.005)


def method_advantage(train_majority_acc: float, train_bce: float, q_bce: float,
                     dev_acc: float, dev_bce: float,
                     d_dev_acc: float, d_dev_bce: float) -> bool:
    """Non-D method advantage: ability plus >=.005 BCE over same-stage D with ACC nondecrease."""
    return (capability_gate(train_majority_acc, train_bce, q_bce, dev_acc, dev_bce)
            and d_dev_bce - dev_bce >= 0.005 and dev_acc >= d_dev_acc)


def n1_null_alarm(null_dev_acc: float, null_train_majority_acc: float,
                  null_dev_bce: float, null_train_bce: float) -> bool:
    """Fixed N1 null alarm vs the null arm's own TRAIN constant (not vs R, not q6)."""
    return (null_dev_acc >= null_train_majority_acc + 0.03
            and null_dev_bce <= null_train_bce - 0.005)


def n2_null_alarm(train_majority_acc: float, train_bce: float, q_bce: float,
                  null_dev_acc: float, null_dev_bce: float) -> bool:
    """N2 null alarm: null meets the ability gate on DEV, or the fixed N1-style alarm."""
    return (capability_gate(train_majority_acc, train_bce, q_bce, null_dev_acc, null_dev_bce)
            or n1_null_alarm(null_dev_acc, train_majority_acc, null_dev_bce, train_bce))


def select_n2_candidate(passing_dev_bce: dict[str, float]) -> str:
    """Lowest DEV BCE; ties within 1e-8 prefer fewer params, then fixed D,S,R,F,FR order."""
    best = min(passing_dev_bce.values())
    tied = [m for m, bce in passing_dev_bce.items() if bce <= best + 1e-8]
    order = ("d", "s", "r", "f", "fr")
    return min(tied, key=lambda m: (N2_PARAM_COUNTS[m], order.index(m)))


def start_r10(r8_ability: bool, r8_method: bool, candidate_is_d: bool) -> bool:
    """r10 starts only if r8 passed ability and (is D with ability, or has method advantage)."""
    return r8_ability and (candidate_is_d or r8_method)


def n4_selection(tested: dict[int, dict[str, bool]]) -> tuple[int, str]:
    """Tier priority: highest rounds with method advantage; else highest with ability."""
    method_rounds = [r for r, flags in tested.items() if flags.get("method")]
    if method_rounds:
        return max(method_rounds), "method"
    ability_rounds = [r for r, flags in tested.items() if flags.get("ability")]
    assert ability_rounds, "no ability-qualified rounds cannot enter N4"
    return max(ability_rounds), "ability"


def test_capability_gate_boundaries() -> None:
    assert capability_gate(0.50, 0.700, 0.700, 0.52, 0.695)
    assert not capability_gate(0.50, 0.700, 0.700, 0.5199, 0.695)  # ACC +.02 exact gate
    assert not capability_gate(0.50, 0.700, 0.700, 0.52, 0.6951)  # vs TRAIN BCE -.005
    assert not capability_gate(0.50, 0.700, 0.690, 0.52, 0.6851)  # vs q BCE -.005 required too


def test_method_advantage_boundaries() -> None:
    base = (0.50, 0.700, 0.700, 0.60, 0.600)
    assert method_advantage(*base, d_dev_acc=0.59, d_dev_bce=0.610)
    assert not method_advantage(*base, d_dev_acc=0.601, d_dev_bce=0.610)  # ACC must not drop
    assert not method_advantage(*base, d_dev_acc=0.59, d_dev_bce=0.6049)  # BCE gain >= .005
    assert not method_advantage(0.50, 0.700, 0.700, 0.51, 0.600,
                                d_dev_acc=0.50, d_dev_bce=0.700)  # ability itself failed


def test_n1_and_n2_null_alarm_rules() -> None:
    assert n1_null_alarm(0.53, 0.50, 0.690, 0.700)
    assert not n1_null_alarm(0.53, 0.50, 0.696, 0.700)  # BCE improvement < .005
    assert not n1_null_alarm(0.52, 0.50, 0.690, 0.700)  # ACC gain < .03
    # N2 alarm: null meeting the ability gate also alarms; Z/M-vs-R null differences alone do not.
    assert n2_null_alarm(0.50, 0.700, 0.700, 0.52, 0.695)
    assert n2_null_alarm(0.50, 0.700, 0.700, 0.53, 0.690)
    assert not n2_null_alarm(0.50, 0.700, 0.700, 0.52, 0.698)


def test_n2_selection_tie_rules() -> None:
    assert select_n2_candidate({"d": 0.60, "f": 0.61, "s": 0.62}) == "d"
    assert select_n2_candidate({"f": 0.60, "s": 0.60, "fr": 0.60}) == "s"  # tie: fewer params
    assert select_n2_candidate({"r": 0.60, "fr": 0.60, "d": 0.60}) == "d"  # params equal: D first
    assert select_n2_candidate({"f": 0.60, "fr": 0.60 + 1e-8}) == "f"  # within 1e-8 is a tie
    assert select_n2_candidate({"f": 0.60, "fr": 0.60 + 1e-4}) == "f"  # strictly lower BCE wins
    # 1e-9 apart is a tie: fewer params first, then D,S,R,F,FR order (r and s both 24961).
    assert select_n2_candidate({"s": 0.60, "r": 0.599999999}) == "s"


def test_stage_branch_table() -> None:
    """Expected transitions; the real controller is manual and follows the spec text."""
    scenarios: list[tuple[dict[str, object], str]] = [
        # All N2 candidates fail the gate: neural cycle stops, N3/N4 never start.
        ({"kind": "ladder", "r6_all_fail": True}, "STOP_N3_N4_COMPLETE_REMAINING_FROZEN_ARMS"),
        # Candidate is D: N3/N4 blocks deduplicate D real/null into a single 2-fit block.
        ({"kind": "block", "candidate_is_d": True}, "DEDUP_TO_2_FITS_PER_BLOCK"),
        ({"kind": "block", "candidate_is_d": False}, "CANDIDATE_PLUS_D_4_FITS_PER_BLOCK"),
        # r8 with ability+method (non-D): start r10.
        ({"kind": "ladder", "r8_ability": True, "r8_method": True,
          "candidate_is_d": False}, "START_R10"),
        # D only needs ability; with ability it starts r10.
        ({"kind": "ladder", "r8_ability": True, "r8_method": False,
          "candidate_is_d": True}, "START_R10"),
        # r8 ability but no method advantage, r6 has method advantage: confirm r6, no r10.
        ({"kind": "ladder", "r8_ability": True, "r8_method": False, "candidate_is_d": False,
          "r6_method": True}, "NO_R10_CONFIRM_R6"),
        # Same, but no round ever had method advantage: keep r8 ability tier into N4.
        ({"kind": "ladder", "r8_ability": True, "r8_method": False, "candidate_is_d": False,
          "r6_method": False}, "NO_R10_KEEP_R8_ABILITY"),
        # r8 ability failed: no r10; confirm the highest qualified round instead.
        ({"kind": "ladder", "r8_ability": False, "r8_method": False, "candidate_is_d": False,
          "r6_ability": True}, "NO_R10_CONFIRM_R6"),
        # Any null alarm: stop, no new science starts.
        ({"kind": "stop", "null_alarm": True}, "ALARM_STOP_AUDIT_DATA_RNG_SPLIT_METRICS"),
        # Missing exit is UNKNOWN: stop new starts, no jumping to higher rounds, no retry.
        ({"kind": "stop", "exit": "missing"}, "UNKNOWN_STOP_MAIN_MODEL_REVIEWS_NO_RETRY"),
        # Shared-code error degrades affected results; no automatic formal retry either.
        ({"kind": "stop", "exit": "shared_code_error"},
         "DEGRADE_AFFECTED_RESULTS_NO_AUTO_RETRY"),
    ]
    for case, expected in scenarios:
        assert _stage_transition(case) == expected, (case, expected)


def _stage_transition(case: dict[str, object]) -> str:
    kind = case.get("kind")
    if kind == "stop":
        if case.get("null_alarm"):
            return "ALARM_STOP_AUDIT_DATA_RNG_SPLIT_METRICS"
        if case.get("exit") == "missing":
            return "UNKNOWN_STOP_MAIN_MODEL_REVIEWS_NO_RETRY"
        assert case.get("exit") == "shared_code_error"
        return "DEGRADE_AFFECTED_RESULTS_NO_AUTO_RETRY"
    if kind == "block":
        return "DEDUP_TO_2_FITS_PER_BLOCK" if case.get("candidate_is_d") \
            else "CANDIDATE_PLUS_D_4_FITS_PER_BLOCK"
    assert kind == "ladder"
    if case.get("r6_all_fail"):
        return "STOP_N3_N4_COMPLETE_REMAINING_FROZEN_ARMS"
    return _stage_transition_inner(case)


def _stage_transition_inner(case: dict[str, object]) -> str:
    candidate_is_d = bool(case.get("candidate_is_d"))
    if start_r10(bool(case.get("r8_ability")), bool(case.get("r8_method")), candidate_is_d):
        return "START_R10"
    if bool(case.get("r6_method")):
        return "NO_R10_CONFIRM_R6"
    if bool(case.get("r8_ability")):
        return "NO_R10_KEEP_R8_ABILITY"
    if bool(case.get("r6_ability")):
        return "NO_R10_CONFIRM_R6"
    return "STOP_N3_N4_COMPLETE_REMAINING_FROZEN_ARMS"


def test_n4_tier_priority_table() -> None:
    # Highest method-advantage round wins even if a higher round has ability only.
    assert n4_selection({6: {"ability": True, "method": True},
                         8: {"ability": True, "method": False}}) == (6, "method")
    # No method advantage anywhere: highest ability round, prediction-only confirmation.
    assert n4_selection({6: {"ability": True, "method": False},
                         8: {"ability": True, "method": False}}) == (8, "ability")
    # r10 tested and method-qualified: it wins over lower method rounds.
    assert n4_selection({6: {"ability": True, "method": True},
                         8: {"ability": True, "method": True},
                         10: {"ability": True, "method": True}}) == (10, "method")
    # r10 failed ability: highest remaining tier wins (ability-only r8 here).
    assert n4_selection({6: {"ability": True, "method": False},
                         8: {"ability": True, "method": False},
                         10: {"ability": False, "method": False}}) == (8, "ability")
