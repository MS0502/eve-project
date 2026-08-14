[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Capture')]
    [string]$Action,
    [Parameter(Mandatory = $true)]
    [string]$Output
)

$ErrorActionPreference = 'Stop'

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-RegistryValue([string]$Path, [string]$Name) {
    try {
        return (Get-ItemProperty -LiteralPath $Path -Name $Name -ErrorAction Stop).$Name
    } catch {
        return $null
    }
}

function Get-RawHostState {
    $defenderStatus = Get-MpComputerStatus -ErrorAction SilentlyContinue
    $defenderPreference = Get-MpPreference -ErrorAction SilentlyContinue
    $os = Get-CimInstance Win32_OperatingSystem
    $computer = Get-CimInstance Win32_ComputerSystem
    $enclosure = Get-CimInstance Win32_SystemEnclosure
    return [ordered]@{
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        elevated = Test-Administrator
        os = [ordered]@{
            caption = $os.Caption
            version = $os.Version
            build_number = $os.BuildNumber
            last_boot_time = $os.LastBootUpTime.ToUniversalTime().ToString('o')
            manufacturer = $computer.Manufacturer
            model = $computer.Model
            chassis_types = @($enclosure.ChassisTypes)
        }
        windows_update = [ordered]@{
            no_auto_reboot_with_logged_on_users = Get-RegistryValue 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' 'NoAutoRebootWithLoggedOnUsers'
            always_auto_reboot_at_scheduled_time = Get-RegistryValue 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' 'AlwaysAutoRebootAtScheduledTime'
            au_options = Get-RegistryValue 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' 'AUOptions'
            target_release_version = Get-RegistryValue 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate' 'TargetReleaseVersion'
            target_release_version_info = Get-RegistryValue 'HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate' 'TargetReleaseVersionInfo'
            cbs_reboot_pending = Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending'
            wu_reboot_required = Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired'
            pending_file_rename = $null -ne (Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager' -Name PendingFileRenameOperations -ErrorAction SilentlyContinue).PendingFileRenameOperations
        }
        power = [ordered]@{
            active_scheme = (& powercfg.exe /getactivescheme 2>&1 | Out-String).Trim()
            sleep = (& powercfg.exe /query SCHEME_CURRENT SUB_SLEEP 2>&1 | Out-String).Trim()
            disk = (& powercfg.exe /query SCHEME_CURRENT SUB_DISK DISKIDLE 2>&1 | Out-String).Trim()
            lid = (& powercfg.exe /qh SCHEME_CURRENT SUB_BUTTONS LIDACTION 2>&1 | Out-String).Trim()
            available_sleep_states = (& powercfg.exe /availablesleepstates 2>&1 | Out-String).Trim()
            hiberboot_enabled = Get-RegistryValue 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' 'HiberbootEnabled'
        }
        defender = [ordered]@{
            antivirus_enabled = $defenderStatus.AntivirusEnabled
            real_time_protection_enabled = $defenderStatus.RealTimeProtectionEnabled
            behavior_monitor_enabled = $defenderStatus.BehaviorMonitorEnabled
            ioav_protection_enabled = $defenderStatus.IoavProtectionEnabled
            signature_last_updated = if ($null -eq $defenderStatus.AntivirusSignatureLastUpdated) { $null } else { $defenderStatus.AntivirusSignatureLastUpdated.ToUniversalTime().ToString('o') }
            disable_realtime_monitoring = $defenderPreference.DisableRealtimeMonitoring
            exclusion_path = @($defenderPreference.ExclusionPath)
            exclusion_process = @($defenderPreference.ExclusionProcess)
            exclusion_extension = @($defenderPreference.ExclusionExtension)
        }
    }
}

$outputPath = [IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $outputPath) {
    throw "Refusing to overwrite host policy record: $outputPath"
}
$parent = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$before = Get-RawHostState
$changes = @()
$after = Get-RawHostState
$packet = [ordered]@{
    schema = 'eve.b5-windows-host-policy-record.v1'
    action = $Action
    mutation_permitted = $false
    manual_only = [ordered]@{
        windows_update_policy = 'Minseok reviews and changes this only in the Windows GUI; NoAutoRebootWithLoggedOnUsers and related values are capture-only here.'
        defender_exclusions = 'Minseok reviews and changes exclusions only in Windows Security; no B5 exclusion is added by this script.'
        plugged_in_power_and_lid = 'Minseok reviews AC sleep, hibernate, hard-disk idle, and lid settings in the Windows GUI.'
    }
    privileged_script = 'scripts/operator/b5_windows_service.ps1'
    before = $before
    changes = $changes
    after = $after
}
[IO.File]::WriteAllText(
    $outputPath,
    ($packet | ConvertTo-Json -Depth 12) + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)
Write-Output $outputPath
