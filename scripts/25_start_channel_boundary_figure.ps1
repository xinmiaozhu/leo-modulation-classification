# Start/resume the complete seven-curve run; plotting is automatic on success.
param([switch]$ExpandedTest)
$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentOutput = Join-Path $experimentRoot 'outputs\channel_boundary_flat_multipath'
$experimentArguments = @('-u', 'scripts/24_run_channel_boundary_figure.py')
if ($ExpandedTest) {
    $experimentOutput = Join-Path $experimentRoot 'outputs\channel_boundary_flat_multipath_test1000'
    $experimentArguments += @('--config', 'configs/experiment/channel_boundary_figure_test1000.json',
        '--output-dir', 'outputs/channel_boundary_flat_multipath_test1000',
        '--checkpoint-source', 'outputs/channel_boundary_flat_multipath')
}
$experimentPython = (Resolve-Path -LiteralPath (Join-Path $experimentRoot '.venv-cuda\Scripts\python.exe')).Path
New-Item -ItemType Directory -Path $experimentOutput -Force | Out-Null
$launcherRecord = Join-Path $experimentOutput 'launcher.json'
if (Test-Path -LiteralPath $launcherRecord) {
    $previousRun = Get-Content -LiteralPath $launcherRecord -Raw | ConvertFrom-Json
    $existingProcess = Get-Process -Id $previousRun.pid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.StartTime.ToUniversalTime().ToString('o') -eq $previousRun.process_started_utc) {
        throw "Figure experiment is already running as PID $($previousRun.pid)."
    }
}
$experimentProcess = Start-Process -FilePath $experimentPython `
    -ArgumentList $experimentArguments `
    -WorkingDirectory $experimentRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $experimentOutput 'runner.stdout.log') `
    -RedirectStandardError (Join-Path $experimentOutput 'runner.stderr.log')
@{
    pid = $experimentProcess.Id
    process_started_utc = $experimentProcess.StartTime.ToUniversalTime().ToString('o')
    output = $experimentOutput
} | ConvertTo-Json | Set-Content -LiteralPath $launcherRecord -Encoding UTF8
Write-Output "Started figure experiment: PID $($experimentProcess.Id)"
