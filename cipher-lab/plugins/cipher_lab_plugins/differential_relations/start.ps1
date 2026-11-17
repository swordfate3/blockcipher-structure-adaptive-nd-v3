param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$Id,
    [Parameter(Mandatory=$true)][string]$GpuUuid,
    [Parameter(Mandatory=$true)][string]$GpuPci
)
$ErrorActionPreference = 'Stop'
$Python = [IO.Path]::GetFullPath($Python)
$Repo = [IO.Path]::GetFullPath($Repo)
$Config = [IO.Path]::GetFullPath($Config)
function Assert-NoReparse([string]$Path) {
    $current = [IO.Path]::GetFullPath($Path)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'UNSAFE_PATH' }
        }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ($parent -eq $current) { break }
        $current = $parent
    }
}
# Existing allowed workspace only; no alternate drive or prefix lookalike.
$allowed = [IO.Path]::GetFullPath('G:\lxy').TrimEnd('\')
Assert-NoReparse $allowed
if (!(Test-Path -LiteralPath $allowed -PathType Container)) { throw 'WORKSPACE_MISSING' }
$Root = [IO.Path]::GetFullPath($Root).TrimEnd('\')
if (!($Root.Equals($allowed, [StringComparison]::OrdinalIgnoreCase)) -and
    !($Root.StartsWith($allowed + '\', [StringComparison]::OrdinalIgnoreCase))) {
    throw 'ROOT_OUTSIDE_ALLOWED_WORKSPACE'
}
foreach ($path in @($Python,$Repo,$Config,$Root)) { Assert-NoReparse $path }
foreach ($relative in @('src','plugins','src/cipher_lab','src/cipher_lab/cli.py',
                         'plugins/cipher_lab_plugins','plugins/cipher_lab_plugins/differential_relations')) {
    Assert-NoReparse (Join-Path $Repo $relative)
}
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'PYTHON_MISSING' }
if (!(Test-Path -LiteralPath $Config -PathType Leaf)) { throw 'CONFIG_MISSING' }
if (!(Test-Path -LiteralPath (Join-Path $Repo 'src/cipher_lab/cli.py') -PathType Leaf)) { throw 'REPO_MISSING' }
if ($Id -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$') { throw 'INVALID_ID' }
if ($GpuUuid -notmatch '^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$') { throw 'INVALID_UUID' }
if ($GpuPci -notmatch '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$') { throw 'INVALID_PCI' }
foreach ($kind in @('data','models','result')) {
    $target = Join-Path (Join-Path $Root $kind) $Id
    Assert-NoReparse $target
    if (Test-Path -LiteralPath $target) { throw 'OCCUPIED' }
}
if (!(Test-Path -LiteralPath $Root)) { [IO.Directory]::CreateDirectory($Root) | Out-Null }
if (!(Test-Path -LiteralPath $Root -PathType Container)) { throw 'ROOT_NOT_DIRECTORY' }
Assert-NoReparse $Root
$work = Join-Path $Root ('.relations-work-' + [Guid]::NewGuid().ToString('N'))
# New-Item is exclusive: never adopt or delete another task's directory.
New-Item -ItemType Directory -Path $work | Out-Null
$oldTmp = $env:TMP
$oldTemp = $env:TEMP
$oldCache = $env:CUDA_CACHE_PATH
$oldTorchHome = $env:TORCH_HOME
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE
$oldPath = $env:PYTHONPATH
$oldDevices = $env:CUDA_VISIBLE_DEVICES
$oldLocation = Get-Location
$code = 4
try {
    $env:TMP = $work
    $env:TEMP = $work
    $env:CUDA_CACHE_PATH = $work
    $env:TORCH_HOME = $work
    $env:PYTHONDONTWRITEBYTECODE = '1'
    Set-Location -LiteralPath $work
    & $Python -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 2)"
    if ($LASTEXITCODE -ne 0) { throw 'PYTHON_REQUIRES_3_12' }
    $env:PYTHONPATH = (Join-Path $Repo 'src') + [IO.Path]::PathSeparator + (Join-Path $Repo 'plugins')
    $env:CUDA_VISIBLE_DEVICES = $GpuUuid.ToUpperInvariant()
    & $Python -m cipher_lab.cli run --plugin cipher_lab_plugins.differential_relations.cli --config $Config --root $Root --id $Id --device cuda:0 --gpu-uuid $env:CUDA_VISIBLE_DEVICES --gpu-pci $GpuPci
    $code = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $oldLocation.Path
    $env:TMP = $oldTmp
    $env:TEMP = $oldTemp
    $env:CUDA_CACHE_PATH = $oldCache
    $env:TORCH_HOME = $oldTorchHome
    $env:PYTHONDONTWRITEBYTECODE = $oldBytecode
    $env:PYTHONPATH = $oldPath
    $env:CUDA_VISIBLE_DEVICES = $oldDevices
    # Only this exact owned subtree. Refuse cleanup through an unexpected reparse.
    Assert-NoReparse $work
    foreach ($child in Get-ChildItem -LiteralPath $work -Recurse -Force) {
        if ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'UNSAFE_TEMP_CLEANUP' }
    }
    Remove-Item -LiteralPath $work -Recurse -Force
}
exit $code
