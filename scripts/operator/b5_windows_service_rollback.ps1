[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RollbackState,
    [switch]$Apply,
    [string]$ExpectedPlanSha256
)

$ErrorActionPreference = 'Stop'
$FastStartupRegistryPath = 'HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power'

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Sha256FromText([string]$Text) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
    $hash = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($hash.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant()
    } finally {
        $hash.Dispose()
    }
}

function Get-FileSha256OrNull([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-ServiceState([string]$Name) {
    $service = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if ($null -eq $service) {
        return [ordered]@{ exists = $false; status = $null; start_type = $null }
    }
    return [ordered]@{
        exists = $true
        status = [string]$service.Status
        start_type = [string]$service.StartType
    }
}

function Get-FastStartupState {
    try {
        $key = Get-Item -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power' -ErrorAction Stop
        $valueExists = @($key.GetValueNames()) -contains 'HiberbootEnabled'
        return [ordered]@{
            read_status = 'RESOLVED'
            value_exists = $valueExists
            value = if ($valueExists) { [int]$key.GetValue('HiberbootEnabled') } else { $null }
            error = $null
        }
    } catch {
        return [ordered]@{
            read_status = 'UNRESOLVED'
            value_exists = $null
            value = $null
            error = $_.Exception.Message
        }
    }
}

function Get-UnsignedPayloadSha256([object]$Payload) {
    $unsigned = [ordered]@{}
    foreach ($property in $Payload.PSObject.Properties) {
        if ($property.Name -ne 'payload_sha256') {
            $unsigned[$property.Name] = $property.Value
        }
    }
    return Get-Sha256FromText ($unsigned | ConvertTo-Json -Depth 20 -Compress)
}

function Add-PayloadSha256([System.Collections.IDictionary]$Payload) {
    $Payload['payload_sha256'] = Get-Sha256FromText ($Payload | ConvertTo-Json -Depth 20 -Compress)
    return $Payload
}

$statePath = [IO.Path]::GetFullPath($RollbackState)
if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
    throw "Rollback state is absent: $statePath"
}
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
if ($state.schema -ne 'eve.b5-windows-privileged-rollback-state.v1') {
    throw 'Rollback state schema differs.'
}
$stateDigest = Get-UnsignedPayloadSha256 $state
if ($stateDigest -ne $state.payload_sha256) {
    throw 'Rollback state digest differs.'
}
if ($state.automatic_rollback_permitted -ne $false) {
    throw 'Rollback state does not prohibit automatic rollback.'
}

$serviceName = [string]$state.service_name
$wrapper = [IO.Path]::GetFullPath([string]$state.wrapper)
$configuration = [IO.Path]::GetFullPath([string]$state.configuration)
$serviceCurrent = Get-ServiceState $serviceName
$fastStartupCurrent = Get-FastStartupState
$wrapperHash = Get-FileSha256OrNull $wrapper
$configurationHash = Get-FileSha256OrNull $configuration
$operations = @()
$blockedReasons = @()

if ($state.source_action -eq 'Install') {
    if ($state.fast_startup_changed -eq $true -and $fastStartupCurrent.read_status -ne 'RESOLVED') {
        $blockedReasons += 'Current Fast Startup state is UNRESOLVED; registry rollback is not authorized.'
    }
    $expectedWrapperHash = [string]$state.expected_created_files.wrapper_sha256
    $expectedConfigurationHash = [string]$state.expected_created_files.configuration_sha256
    if ($serviceCurrent.exists) {
        if ($wrapperHash -ne $expectedWrapperHash -or $configurationHash -ne $expectedConfigurationHash) {
            $blockedReasons += 'Installed service exists but its WinSW executable or configuration digest drifted; no service deletion is authorized.'
        } else {
            if ($serviceCurrent.status -ne 'Stopped') {
                $operations += "& '$wrapper' stopwait"
            }
            $operations += "& '$wrapper' uninstall"
        }
    }
    if ($null -ne $wrapperHash) {
        if ($wrapperHash -eq $expectedWrapperHash) {
            $operations += "Remove-Item -LiteralPath '$wrapper'"
        } else {
            $blockedReasons += 'WinSW executable digest drifted; file removal is not authorized.'
        }
    }
    if ($null -ne $configurationHash) {
        if ($configurationHash -eq $expectedConfigurationHash) {
            $operations += "Remove-Item -LiteralPath '$configuration'"
        } else {
            $blockedReasons += 'WinSW configuration digest drifted; file removal is not authorized.'
        }
    }
    if ($state.fast_startup_changed -eq $true) {
        if ($state.before.fast_startup.value_exists) {
            $operations += "reg.exe add `"$FastStartupRegistryPath`" /v HiberbootEnabled /t REG_DWORD /d $([int]$state.before.fast_startup.value) /f"
        } else {
            $operations += "reg.exe delete `"$FastStartupRegistryPath`" /v HiberbootEnabled /f"
        }
    }
} else {
    if (-not $serviceCurrent.exists) {
        $blockedReasons += 'Service disappeared after the reviewed control action; prior status cannot be restored.'
    } else {
        $priorStatus = [string]$state.before.service.status
        if ($priorStatus -eq 'Running' -and $serviceCurrent.status -ne 'Running') {
            $operations += "& '$wrapper' start"
        } elseif ($priorStatus -eq 'Stopped' -and $serviceCurrent.status -ne 'Stopped') {
            $operations += "& '$wrapper' stopwait"
        } elseif ($priorStatus -notin @('Running', 'Stopped')) {
            $blockedReasons += "Unsupported prior service status: $priorStatus"
        }
    }
}

$plan = [ordered]@{
    schema = 'eve.b5-windows-privileged-rollback-plan.v1'
    dry_run_default = $true
    rollback_state = $statePath
    rollback_state_payload_sha256 = $state.payload_sha256
    source_plan_sha256 = $state.source_plan_sha256
    source_action = $state.source_action
    before_rollback = [ordered]@{
        service = $serviceCurrent
        fast_startup = $fastStartupCurrent
        wrapper_sha256 = $wrapperHash
        configuration_sha256 = $configurationHash
    }
    exact_operations = $operations
    blocked_reasons = $blockedReasons
    retained = @(
        'WinSW logs and physical evidence are not recursively deleted.',
        'Microsoft Defender and Windows Update settings are not changed.',
        'authority_active_for_runtime and t=0 establishment data are not changed.'
    )
}
$plan = Add-PayloadSha256 $plan
Write-Output ($plan | ConvertTo-Json -Depth 20)
Write-Output "PLAN_SHA256=$($plan.payload_sha256)"

if (-not $Apply) {
    Write-Output 'DRY-RUN ONLY: no files, registry values, services, policies, or runtime state were changed.'
    exit 0
}
if (-not (Test-Administrator)) {
    throw 'B5 privileged rollback requires an elevated administrator token.'
}
if ($ExpectedPlanSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'Apply requires -ExpectedPlanSha256 from the reviewed rollback dry-run output.'
}
if ($ExpectedPlanSha256.ToLowerInvariant() -ne $plan.payload_sha256) {
    throw 'Current rollback plan differs from the reviewed plan; refusing Apply.'
}
if ($blockedReasons.Count -ne 0) {
    throw 'Rollback is blocked by drift; review a new plan before any privileged action.'
}

foreach ($operation in $operations) {
    $externalExitCode = $null
    if ($operation -like "& '* stopwait") {
        & $wrapper stopwait
        $externalExitCode = $LASTEXITCODE
    } elseif ($operation -like "& '* uninstall") {
        & $wrapper uninstall
        $externalExitCode = $LASTEXITCODE
    } elseif ($operation -like "& '* start") {
        & $wrapper start
        $externalExitCode = $LASTEXITCODE
    } elseif ($operation -like "Remove-Item*'$wrapper'") {
        Remove-Item -LiteralPath $wrapper
    } elseif ($operation -like "Remove-Item*'$configuration'") {
        Remove-Item -LiteralPath $configuration
    } elseif ($operation -like 'reg.exe add*') {
        & reg.exe add $FastStartupRegistryPath /v HiberbootEnabled /t REG_DWORD /d ([int]$state.before.fast_startup.value) /f | Out-Null
        $externalExitCode = $LASTEXITCODE
    } elseif ($operation -like 'reg.exe delete*') {
        & reg.exe delete $FastStartupRegistryPath /v HiberbootEnabled /f | Out-Null
        $externalExitCode = $LASTEXITCODE
    } else {
        throw "Unrecognized reviewed rollback operation: $operation"
    }
    if ($null -ne $externalExitCode -and $externalExitCode -ne 0) {
        throw "Rollback operation failed with exit code ${externalExitCode}: $operation"
    }
}

Write-Output 'ROLLBACK COMPLETED. Evidence logs were retained; runtime authority remains false and t=0 was not started.'
