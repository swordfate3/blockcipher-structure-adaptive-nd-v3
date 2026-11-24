# Run in a dedicated PowerShell process, after the controller's exact checkout.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('\A[A-Za-z0-9][A-Za-z0-9_-]{0,63}\z')]
    [string]$RunId,
    [ValidateSet('direct-block-cube-sharedm-simon7.json',
                 'direct-block-cube-sharedm-simon7-50epoch.json',
                 'direct-block-cube-lstm-simon7.json',
                 'direct-block-cube-lstm-simon-r1r3.json')]
    [string]$ConfigName = 'direct-block-cube-sharedm-simon7.json',
    [string]$SourceRoot
)
$ErrorActionPreference = 'Stop'
$base = 'G:\lxy\shared-m-cipher-nd-runs'
$runRoot = Join-Path $base $RunId
$lab = Join-Path $runRoot 'release\cipher-lab'
$work = Join-Path $runRoot 'work'
$python = 'C:\Python313\python.exe'
$config = Join-Path (Join-Path $lab 'configs') $ConfigName
$reuse = ($ConfigName -ieq 'direct-block-cube-sharedm-simon7-50epoch.json' -or
          $ConfigName -ieq 'direct-block-cube-lstm-simon7.json' -or
          $ConfigName -ieq 'direct-block-cube-lstm-simon-r1r3.json')
if ($reuse -ne (-not [string]::IsNullOrWhiteSpace($SourceRoot))) { throw 'INVALID_SOURCE_ROOT' }
# The launcher changes directory before running Python; require an explicit absolute path.
if ($reuse -and $SourceRoot -notmatch '\A(?:[A-Za-z]:[\\/]|\\\\)') {
    throw 'INVALID_SOURCE_ROOT'
}
$uuid = 'GPU-f80e8056-5b1a-a0c7-9b34-fb86b20206d5'

# Use exactly the script in this run's own checkout; do not adopt old work roots.
if ($PSScriptRoot -ine (Join-Path $lab 'scripts')) { throw 'WRONG_CHECKOUT' }
foreach ($dir in @('G:\lxy', $base, $runRoot, (Join-Path $runRoot 'release'), $lab)) {
    $item = Get-Item -LiteralPath $dir
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'UNSAFE_DIRECTORY'
    }
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf) -or
    -not (Test-Path -LiteralPath $config -PathType Leaf)) { throw 'MISSING_RELEASE_INPUT' }
if ($env:CUDA_VISIBLE_DEVICES -and ($env:CUDA_VISIBLE_DEVICES -cne $uuid)) {
    throw 'GPU_IDENTITY'
}
New-Item -ItemType Directory -Path $work -ErrorAction Stop | Out-Null
foreach ($dir in @('temp', 'cache')) {
    New-Item -ItemType Directory -Path (Join-Path $work $dir) | Out-Null
}
$env:CUDA_VISIBLE_DEVICES = $uuid
$env:TEMP = Join-Path $work 'temp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:XDG_CACHE_HOME = Join-Path $work 'cache'
$env:TORCH_HOME = Join-Path $env:XDG_CACHE_HOME 'torch'
$env:TORCH_EXTENSIONS_DIR = Join-Path $env:XDG_CACHE_HOME 'torch-extensions'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $env:XDG_CACHE_HOME 'torchinductor'
$env:TRITON_CACHE_DIR = Join-Path $env:XDG_CACHE_HOME 'triton'
$env:CUDA_CACHE_PATH = Join-Path $env:XDG_CACHE_HOME 'cuda'
$env:CUDA_CACHE_DISABLE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Join-Path $lab 'src') + ';' + (Join-Path $lab 'plugins')
Remove-Item Env:PYTHONNOUSERSITE, Env:PYTHONUSERBASE, Env:PYTHONHOME -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
# PS 7.3+ can promote a nonzero native status to an exception. Preserve the code.
$PSNativeCommandUseErrorActionPreference = $false
Set-Location -LiteralPath $work
$wallSeconds = if ($reuse) { 10800 } else { 7200 }
$deadline = [DateTime]::UtcNow.AddSeconds($wallSeconds).ToString(
    "yyyy-MM-ddTHH:mm:ss.ffffff'Z'", [Globalization.CultureInfo]::InvariantCulture)
$cliArgs = @('-B', '-m', 'cipher_lab', 'run', '--plugin',
    'cipher_lab_plugins.direct_block.cube_compare.cli', '--config', $config,
    '--root', $work, '--run-id', $RunId, '--deadline-utc', $deadline)
if ($reuse) { $cliArgs += @('--source-root', $SourceRoot) }
$stdout = Join-Path $work 'stdout.log'
$stderr = Join-Path $work 'stderr.log'
# Windows PowerShell may treat redirected native stderr as error records.
$ErrorActionPreference = 'Continue'
# PS5.1 writes the native status globally; a local null shadows it in a function.
$global:LASTEXITCODE = $null
& $python @cliArgs 1> $stdout 2> $stderr
$code = $global:LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($null -eq $code) { throw 'NATIVE_EXIT_UNKNOWN' }
try {
    Set-Content -LiteralPath (Join-Path $work 'exit.txt') -Value $code -Encoding ascii -ErrorAction Stop
} catch {
    Write-Warning 'EXIT_EVIDENCE_WRITE_FAILED' -WarningAction Continue
    if ($code -eq 0) { exit 1 }
}
exit $code
