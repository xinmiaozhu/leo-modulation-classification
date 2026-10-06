# Start/resume the full experiment in a hidden process with persistent logs.
$ErrorActionPreference = 'Stop'
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$experimentOutput = Join-Path $experimentRoot 'outputs\channel_boundary_full'
$experimentPython = (Resolve-Path -LiteralPath (Join-Path $experimentRoot '.venv-cuda\Scripts\python.exe')).Path
New-Item -ItemType Directory -Path $experimentOutput -Force | Out-Null
$launcherRecord = Join-Path $experimentOutput 'launcher.json'
if (Test-Path -LiteralPath $launcherRecord) {
    $previousRun = Get-Content -LiteralPath $launcherRecord -Raw | ConvertFrom-Json
    $existingProcess = Get-Process -Id $previousRun.pid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.StartTime.ToUniversalTime().ToString('o') -eq $previousRun.process_started_utc) {
        throw "Experiment is already running as PID $($previousRun.pid)."
    }
}
$experimentArguments = @(
    '-u', 'scripts/22_run_channel_boundary.py',
    '--config', 'configs/experiment/channel_boundary_full.json',
    '--output-dir', 'outputs/channel_boundary_full', '--device', 'cuda'
)
$experimentProcess = Start-Process -FilePath $experimentPython -ArgumentList $experimentArguments `
    -WorkingDirectory $experimentRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $experimentOutput 'runner.stdout.log') `
    -RedirectStandardError (Join-Path $experimentOutput 'runner.stderr.log')
@{
    pid = $experimentProcess.Id
    process_started_utc = $experimentProcess.StartTime.ToUniversalTime().ToString('o')
    python = $experimentPython
    arguments = $experimentArguments
    output = $experimentOutput
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launcherRecord -Encoding UTF8
Write-Output "Started full channel-boundary experiment: PID $($experimentProcess.Id)"
Write-Output "Progress: $(Join-Path $experimentOutput 'run_progress.json')"
