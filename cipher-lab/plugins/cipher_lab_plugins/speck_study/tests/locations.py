"""Source locations independent of pytest invocation cwd or repository name."""

from cipher_lab_plugins.speck_study.paths import source_roots

CORE_ROOT, PLUGIN_ROOT = source_roots()
ROOT = CORE_ROOT.parent
PLUGIN = PLUGIN_ROOT/'cipher_lab_plugins'/'speck_study'
RUNTIME = PLUGIN/'runtime'
