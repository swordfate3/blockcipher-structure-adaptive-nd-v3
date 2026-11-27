"""Source identity for the card's explicit production path, not old runners."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from cipher_lab import artifacts
from cipher_lab.artifacts import ArtifactError, _source

from ..source import no_reparse

_CORE = {
    '': ('__init__', '__main__', 'artifacts', 'cli', 'config', 'execution', 'plugin_cli',
         'training_config'),
    'ciphers': ('__init__', 'present', 'present_prefix'),
    'models': ('__init__', 'feature_mlp', 'frontier_mlp', 'mlp'),
    'models/dedicated': ('__init__', 'feistel', 'present', 'skinny'),
    'models/shared_m': ('__init__', 'components', 'router', 'spec', 'system'),
    'representations': ('__init__', 'dedicated', 'input', 'neighborhoods', 'tasks'),
    'training': ('__init__', 'early_stopping'),
}
_PLUGINS = {
    '': ('__init__',),
    'direct_block': ('__init__', 'fit'),
    'direct_block/cube_compare': ('__init__', 'config', 'preflight', 'resources'),
    'innovation1': ('__init__', 'batch', 'checkpoint', 'config', 'contracts', 'models', 'source'),
    'innovation1/four_single': ('__init__', 'models'),
    'innovation1/layout_probe': ('__init__', 'layout', 'models'),
    'innovation1/mask_aux': ('__init__', 'config', 'kat'),
    'innovation1/mask_aux_single': ('__init__', 'config', 'data'),
    'innovation1/prefix_bridge': ('__init__', 'cli', 'config', 'data', 'kat', 'metrics', 'model',
                                  'runtime', 'training', 'weights'),
    'innovation1/s2_window': ('__init__', 'config', 'training'),
    'innovation1/single_pair': ('__init__', 'models'),
    'innovation1/single_train': ('__init__', 'config', 'weights'),
    'innovation1/present_card_fresh': ('__init__', 'cli', 'config', 'data', 'layout', 'model',
                                       'present_public', 'report', 'runtime', 'source',
                                       'training', 'weights'),
}


def source_identity() -> dict[str, str]:
    core = Path(artifacts.__file__).absolute().parent
    plugins = Path(__file__).absolute().parents[2]
    result: dict[str, str] = {}
    try:
        for root, inventory in ((core, _CORE), (plugins, _PLUGINS)):
            for sub, modules in inventory.items():
                for module in modules:
                    rel = Path(sub, f'{module}.py')
                    result[f'{root.name}/{rel.as_posix()}'] = sha256(
                        no_reparse(root / rel).read_bytes()).hexdigest()
        _source(result)
    except (OSError, ValueError):
        raise ArtifactError('SOURCE_UNAVAILABLE') from None
    return dict(sorted(result.items()))
