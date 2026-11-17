from pathlib import Path


def test_single_script_explicit_binding_and_no_installation() -> None:
    p = Path("plugins/cipher_lab_plugins/differential_relations/start.ps1")
    assert p.is_file(), "single startup script missing"
    text = p.read_text()
    for token in (
        "3,12",
        "CUDA_VISIBLE_DEVICES",
        "ToUpperInvariant",
        "gpu-pci",
        "gpu-uuid",
        "cipher_lab.cli",
        "--plugin",
        "ReparsePoint",
        "PYTHONPATH",
    ):
        assert token in text
    for token in ("pip install", "ssh ", "git ", "Start-Process", "nvidia-smi"):
        assert token not in text


def test_start_bounds_all_new_files_and_cleans_owned_work() -> None:
    text = Path("plugins/cipher_lab_plugins/differential_relations/start.ps1").read_text()
    for token in (
        "G:\\lxy",
        "StartsWith",
        "OrdinalIgnoreCase",
        "$env:TMP",
        "$env:TEMP",
        "finally",
        "Remove-Item -LiteralPath $work",
        "PYTHONDONTWRITEBYTECODE",
    ):
        assert token in text
    assert "Copy-Item" not in text


def test_start_redirects_driver_cache_to_owned_work() -> None:
    text = Path("plugins/cipher_lab_plugins/differential_relations/start.ps1").read_text()
    assert "$env:CUDA_CACHE_PATH = $work" in text
    assert "$env:TORCH_HOME = $work" in text
