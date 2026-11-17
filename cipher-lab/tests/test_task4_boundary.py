"""Static migration contracts; never run science."""
from pathlib import Path

import pytest

from cipher_lab.config import load_experiment


@pytest.mark.parametrize('name', ['fourier-example.json', 'integral-search.json',
                                   'speck-study-example.json'])
def test_base_rejects_external_method(name: str, tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    path = project / 'configs' / name
    with pytest.raises(ValueError, match='unsupported experiment'):
        load_experiment(path)


def test_search_plugin_static_entry() -> None:
    from cipher_lab_plugins.integral_search.cli import _estimate
    from cipher_lab_plugins.integral_search.config import load_experiment

    project = Path(__file__).resolve().parents[1]
    counts = _estimate(load_experiment(project / 'configs/integral-search-k18k20.json'))
    assert counts['planned_real_blocks'] == 4227858432


@pytest.mark.parametrize('command', ['check', 'run'])
@pytest.mark.parametrize('name', ['fourier-example.json', 'integral-search.json',
                                'speck-study-example.json'])
def test_external_method_rejected_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, name: str,
) -> None:
    import cipher_lab.cli as cli
    from cipher_lab.artifacts import RunStore

    def claim(*args: object, **kwargs: object) -> None:
        pytest.fail('external method reached base claim')
    monkeypatch.setattr(RunStore, 'claim', claim)
    project = Path(__file__).resolve().parents[1]
    argv = [command, '--config', str(project / 'configs' / name),
            '--policy', str(tmp_path / 'absent-policy.json')]
    if command == 'run':
        argv += ['--root', str(tmp_path / 'out'), '--id', 'rejected', '--backend', 'demo']
    assert cli.main(argv) == 2
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('flag', ['--teacher-python', '--teacher-weights', '--source-manifest',
                                '--weights-revision', '--weights-sha256',
                                '--gpu-uuid', '--gpu-pci'])
def test_base_rejects_experiment_runtime_flags(flag: str, tmp_path: Path) -> None:
    from cipher_lab.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(['run', '--config', 'absent', '--policy', 'absent', '--backend', 'demo',
              '--root', str(tmp_path / 'out'), '--id', 'no-claim', flag, 'untrusted'])
    assert stopped.value.code == 2
    assert not (tmp_path / 'out').exists()
