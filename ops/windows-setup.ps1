# One-time Windows setup when the stack runs in WSL2. Run in an *Administrator* PowerShell:
#   powershell -ExecutionPolicy Bypass -File windows-setup.ps1 [-Distro Ubuntu-24.04] [-NginxDir C:\nginx]
#
#  1. Task "wsl-keepalive": starts the distro at boot and keeps it running
#     (WSL stops an idle distro otherwise). It also re-fires every 5 minutes,
#     so the distro comes back by itself after the 02:00 job reboots it.
#     MultipleInstances=IgnoreNew makes that a no-op while WSL is up.
#  2. Task "wsl-nightly-update" at 03:30: `wsl --update`; if WSL or its kernel
#     changed, `wsl --shutdown` and start the distro again on the new one.
#  3. Optional: tests and reloads a Windows nginx in -NginxDir.
#  4. Tells the Linux side that unattended reboots are now safe.
#  5. Windows Update active hours 08:00-23:00, so Windows restarts overnight.
param(
    [string]$Distro = '',
    [string]$NginxDir = ''
)
$ErrorActionPreference = 'Stop'
if (-not $Distro) {
    $Distro = ((wsl.exe -l -q | Out-String) -replace "`0",'' -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -First 1).Trim()
}
Write-Host "Distro: $Distro"
$cred = Get-Credential -UserName "$env:USERDOMAIN\$env:USERNAME" -Message 'Your Windows password (the scheduled tasks run as you, even when signed out)'
$pw = $cred.GetNetworkCredential().Password
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

# 1 -- keepalive + 5-minute watchdog
$t = Get-ScheduledTask -TaskName 'wsl-keepalive' -ErrorAction SilentlyContinue
$rep = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5)
if (-not $t) {
    $act = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument "-d $Distro --exec /bin/sleep infinity"
    $set.ExecutionTimeLimit = 'PT0S'
    Register-ScheduledTask -TaskName 'wsl-keepalive' -Action $act -Trigger @((New-ScheduledTaskTrigger -AtStartup), $rep) `
        -Settings $set -User $cred.UserName -Password $pw -RunLevel Highest | Out-Null
    Write-Host 'wsl-keepalive: created (at boot + every 5 minutes)'
} else {
    if (-not ($t.Triggers | Where-Object { $_.Repetition.Interval -eq 'PT5M' })) { $t.Triggers += $rep }
    $t.Settings.DisallowStartIfOnBatteries = $false
    $t.Settings.StopIfGoingOnBatteries = $false
    Set-ScheduledTask -InputObject $t -User $cred.UserName -Password $pw | Out-Null
    Write-Host 'wsl-keepalive: now also re-checks every 5 minutes'
}
Start-ScheduledTask -TaskName 'wsl-keepalive'

# 2 -- nightly WSL update
$dir = 'C:\ProgramData\stack'; New-Item -ItemType Directory -Force $dir | Out-Null
@"
`$log = '$dir\wsl-nightly-update.log'
`$before = (wsl.exe --version | Out-String) -replace "`0",''
wsl.exe --update 2>&1 | Out-String | ForEach-Object { `$_ -replace "`0",'' } | Add-Content `$log
`$after = (wsl.exe --version | Out-String) -replace "`0",''
if (`$before -ne `$after) {
    Add-Content `$log "`$(Get-Date -f s) WSL updated, restarting the VM"
    wsl.exe --shutdown
    Start-Sleep 10
    Start-ScheduledTask -TaskName 'wsl-keepalive'
} else { Add-Content `$log "`$(Get-Date -f s) WSL already current" }
"@ | Set-Content "$dir\wsl-nightly-update.ps1"
$act = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File $dir\wsl-nightly-update.ps1"
$set2 = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName 'wsl-nightly-update' -Action $act -Trigger (New-ScheduledTaskTrigger -Daily -At 3:30am) `
    -Settings $set2 -User $cred.UserName -Password $pw -RunLevel Highest -Force | Out-Null
Write-Host 'wsl-nightly-update: daily at 03:30'

# 3 -- optional Windows nginx in front of the stack
if ($NginxDir -and (Test-Path "$NginxDir\nginx.exe")) {
    & "$NginxDir\nginx.exe" -p $NginxDir -t
    if ($LASTEXITCODE -eq 0) { & "$NginxDir\nginx.exe" -p $NginxDir -s reload; Write-Host 'nginx reloaded' }
}

# 4 -- allow reboots from the Linux nightly job
wsl.exe -d $Distro -u root -- sh -c 'mkdir -p /etc/stack-maintenance && touch /etc/stack-maintenance/allow-reboot'
Write-Host 'The Linux nightly job may now reboot the distro when updates need it.'

# 5 -- Windows Update restarts outside the day
Set-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -Name ActiveHoursStart -Value 8 -Type DWord
Set-ItemProperty 'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings' -Name ActiveHoursEnd -Value 23 -Type DWord
Write-Host 'Windows Update active hours: 08:00-23:00'
