[CmdletBinding()]
param(
    [ValidateSet('Install', 'Start', 'Stop', 'Restart')]
    [string]$Action = 'Install',

    [switch]$Apply,
    [string]$ExpectedPlanSha256,
    [string]$RollbackState,

    [string]$ServiceName = 'EveB5Supervisor',
    [string]$DeployDirectory = 'C:\ProgramData\EVE\B5\Service',
    [ValidateSet('Running', 'StoppedLatched')]
    [string]$ExpectedTerminalStatus = 'Running',
    [string]$WinSWPath,
    [string]$WinSWSha256,
    [string]$PythonPath,
    [string]$RepoPath,
    [string]$AuthorityStore,
    [string]$RuntimeReceipt,
    [string]$SentinelPath,
    [string]$AuditLog,
    [string]$AlertLog,
    [string]$StateFile,
    [string]$ControlFile,
    [string]$ChildRawLog,
    [string]$ChildReadyFile
)

$ErrorActionPreference = 'Stop'
$FastStartupRegistryPath = 'HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Power'
$FastStartupPowerShellPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power'

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Assert-Administrator {
    if (-not (Test-Administrator)) {
        throw 'B5 privileged application requires an elevated administrator token.'
    }
}

function Escape-Xml([string]$Value) {
    return [Security.SecurityElement]::Escape($Value)
}

function Quote-Argument([string]$Value) {
    return '&quot;' + (Escape-Xml $Value) + '&quot;'
}

function Resolve-RequiredPath([string]$Value, [string]$Field, [bool]$MustExist = $true) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Field is required for $Action."
    }
    $resolved = [IO.Path]::GetFullPath($Value)
    if ($MustExist -and -not (Test-Path -LiteralPath $resolved)) {
        throw "$Field does not exist: $resolved"
    }
    return $resolved
}

function Get-FileSha256OrNull([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-FastStartupState {
    try {
        $key = Get-Item -LiteralPath $FastStartupPowerShellPath -ErrorAction Stop
        $valueExists = @($key.GetValueNames()) -contains 'HiberbootEnabled'
        return [ordered]@{
            read_status = 'RESOLVED'
            key_exists = $true
            value_exists = $valueExists
            value = if ($valueExists) { [int]$key.GetValue('HiberbootEnabled') } else { $null }
            error = $null
        }
    } catch {
        return [ordered]@{
            read_status = 'UNRESOLVED'
            key_exists = $null
            value_exists = $null
            value = $null
            error = $_.Exception.Message
        }
    }
}

function Get-ServiceState([string]$Name) {
    $service = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if ($null -eq $service) {
        return [ordered]@{
            exists = $false
            status = $null
            start_type = $null
        }
    }
    return [ordered]@{
        exists = $true
        status = [string]$service.Status
        start_type = [string]$service.StartType
    }
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

function Add-PayloadSha256([System.Collections.IDictionary]$Payload) {
    $unsigned = $Payload | ConvertTo-Json -Depth 20 -Compress
    $Payload['payload_sha256'] = Get-Sha256FromText $unsigned
    return $Payload
}

function Write-AtomicJson([string]$Path, [System.Collections.IDictionary]$Payload) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = "$Path.tmp-$([Guid]::NewGuid().ToString('N'))"
    try {
        [IO.File]::WriteAllText(
            $temporary,
            ($Payload | ConvertTo-Json -Depth 20) + [Environment]::NewLine,
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $temporary -Destination $Path
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

$deploy = [IO.Path]::GetFullPath($DeployDirectory)
$wrapper = Join-Path $deploy ($ServiceName + '.exe')
$configuration = Join-Path $deploy ($ServiceName + '.xml')
$serviceBefore = Get-ServiceState $ServiceName
$fastStartupBefore = Get-FastStartupState
$fastStartupChangeRequired = -not (
    $fastStartupBefore.read_status -eq 'RESOLVED' -and
    $fastStartupBefore.value_exists -eq $true -and
    $fastStartupBefore.value -eq 0
)
$resolved = [ordered]@{}
$xml = $null
$xmlSha256 = $null
$exactOperations = @()
$requiredPreconditions = @()

if ($Action -eq 'Install') {
    $resolved.WinSWPath = Resolve-RequiredPath $WinSWPath 'WinSWPath'
    if ($WinSWSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'WinSWSha256 must be an exact SHA-256 digest.'
    }
    $resolved.WinSWSha256 = $WinSWSha256.ToLowerInvariant()
    $actualWinSWHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolved.WinSWPath).Hash.ToLowerInvariant()
    if ($actualWinSWHash -ne $resolved.WinSWSha256) {
        throw "WinSW SHA-256 differs: $actualWinSWHash"
    }
    $resolved.PythonPath = Resolve-RequiredPath $PythonPath 'PythonPath'
    $resolved.RepoPath = Resolve-RequiredPath $RepoPath 'RepoPath'
    $resolved.AuthorityStore = Resolve-RequiredPath $AuthorityStore 'AuthorityStore'
    $resolved.RuntimeReceipt = Resolve-RequiredPath $RuntimeReceipt 'RuntimeReceipt'
    $resolved.SentinelPath = Resolve-RequiredPath $SentinelPath 'SentinelPath' $false
    $resolved.AuditLog = Resolve-RequiredPath $AuditLog 'AuditLog' $false
    $resolved.AlertLog = Resolve-RequiredPath $AlertLog 'AlertLog' $false
    $resolved.StateFile = Resolve-RequiredPath $StateFile 'StateFile' $false
    $resolved.ControlFile = Resolve-RequiredPath $ControlFile 'ControlFile'
    $resolved.ChildRawLog = Resolve-RequiredPath $ChildRawLog 'ChildRawLog' $false
    $resolved.ChildReadyFile = Resolve-RequiredPath $ChildReadyFile 'ChildReadyFile' $false
    $resolved.SupervisorScript = Resolve-RequiredPath (Join-Path $resolved.RepoPath 'scripts\operator\b5_windows_supervisor.py') 'SupervisorScript'
    $resolved.RuntimeProbeScript = Resolve-RequiredPath (Join-Path $resolved.RepoPath 'scripts\operator\b5_runtime_probe.py') 'RuntimeProbeScript'

    $arguments = @(
        (Quote-Argument $resolved.SupervisorScript), 'run',
        '--sentinel', (Quote-Argument $resolved.SentinelPath),
        '--audit-log', (Quote-Argument $resolved.AuditLog),
        '--alert-log', (Quote-Argument $resolved.AlertLog),
        '--state-file', (Quote-Argument $resolved.StateFile),
        '--authority-store', (Quote-Argument $resolved.AuthorityStore),
        '--runtime-receipt', (Quote-Argument $resolved.RuntimeReceipt),
        '--backoff-initial', '1', '--backoff-max', '60', '--',
        (Quote-Argument $resolved.PythonPath), (Quote-Argument $resolved.RuntimeProbeScript),
        '--database', (Quote-Argument $resolved.AuthorityStore),
        '--control', (Quote-Argument $resolved.ControlFile),
        '--raw-log', (Quote-Argument $resolved.ChildRawLog),
        '--ready', (Quote-Argument $resolved.ChildReadyFile)
    ) -join ' '
    $xml = @"
<service>
  <id>$(Escape-Xml $ServiceName)</id>
  <name>$(Escape-Xml $ServiceName)</name>
  <description>EVE B5 exit-aware supervisor. The service wrapper never starts EVE directly.</description>
  <executable>$(Escape-Xml $resolved.PythonPath)</executable>
  <arguments>$arguments</arguments>
  <workingdirectory>$(Escape-Xml $resolved.RepoPath)</workingdirectory>
  <startmode>Automatic</startmode>
  <delayedAutoStart>false</delayedAutoStart>
  <stoptimeout>15 sec</stoptimeout>
  <onfailure action="restart" delay="10 sec" />
  <resetfailure>1 hour</resetfailure>
  <logpath>$(Escape-Xml (Join-Path $deploy 'winsw-logs'))</logpath>
  <log mode="roll-by-size-time">
    <sizeThreshold>10240</sizeThreshold>
    <pattern>yyyyMMdd</pattern>
    <autoRollAtTime>00:00:00</autoRollAtTime>
    <zipOlderThanNumDays>7</zipOlderThanNumDays>
    <zipDateFormat>yyyyMMdd</zipDateFormat>
  </log>
</service>
"@
    $xmlSha256 = Get-Sha256FromText $xml
    $rollback = Resolve-RequiredPath $RollbackState 'RollbackState' $false
    $resolved.RollbackState = $rollback

    $requiredPreconditions = @(
        'The reviewed service name is absent.',
        'The destination WinSW executable and XML configuration are absent.',
        'The rollback-state path is absent.',
        'The supplied WinSW SHA-256, pinned runtime receipt, final repository checkout, proof-store copy, and control file are exact and readable.'
    )
    $exactOperations = @(
        "Write rollback state to '$rollback' before host mutation."
    )
    if ($fastStartupChangeRequired) {
        $exactOperations += "reg.exe add `"$FastStartupRegistryPath`" /v HiberbootEnabled /t REG_DWORD /d 0 /f"
    }
    $exactOperations += @(
        "New-Item -ItemType Directory -Path '$deploy' -Force",
        "Copy-Item -LiteralPath '$($resolved.WinSWPath)' -Destination '$wrapper'",
        "[IO.File]::WriteAllText('$configuration', <reviewed XML>, UTF8-no-BOM); XML SHA-256=$xmlSha256",
        "& '$wrapper' install",
        "sc.exe config '$ServiceName' start= auto",
        "sc.exe failure '$ServiceName' reset= 3600 actions= restart/10000"
    )
} else {
    if (-not (Test-Path -LiteralPath $wrapper -PathType Leaf)) {
        throw "Installed WinSW wrapper is absent: $wrapper"
    }
    $resolved.RollbackState = Resolve-RequiredPath $RollbackState 'RollbackState' $false
    $requiredPreconditions = @(
        'The reviewed WinSW wrapper exists.',
        'The rollback-state path is absent.',
        'The current service state is identical to the state printed in this plan.'
    )
    $exactOperations = @("& '$wrapper' $($Action.ToLowerInvariant())")
}

$plan = [ordered]@{
    schema = 'eve.b5-windows-privileged-plan.v1'
    dry_run_default = $true
    action = $Action
    service_name = $ServiceName
    service_account = 'LocalSystem (WinSW default); this is a privileged runtime identity and is part of the reviewed scope'
    exact_paths = [ordered]@{
        deploy_directory = $deploy
        wrapper = $wrapper
        configuration = $configuration
        rollback_state = $resolved.RollbackState
        resolved_install_inputs = $resolved
    }
    before = [ordered]@{
        fast_startup = $fastStartupBefore
        service = $serviceBefore
        deploy_directory_exists = Test-Path -LiteralPath $deploy
        wrapper_sha256 = Get-FileSha256OrNull $wrapper
        configuration_sha256 = Get-FileSha256OrNull $configuration
    }
    after_required = if ($Action -eq 'Install') {
        [ordered]@{
            hiberboot_enabled = 0
            fast_startup_registry_write_required = $fastStartupChangeRequired
            service_exists = $true
            service_start_type = 'Automatic'
            service_status = 'Stopped until a separately reviewed Start action'
            wrapper_sha256 = $resolved.WinSWSha256
            configuration_sha256 = $xmlSha256
        }
    } else {
        [ordered]@{
            service_status = switch ($Action) {
                'Start' { $ExpectedTerminalStatus }
                'Stop' { 'Stopped' }
                'Restart' { $ExpectedTerminalStatus }
            }
            stopped_latched_definition = 'Service starts the supervisor, which observes an existing sentinel, launches no child, and stops normally.'
            configuration_changed = $false
        }
    }
    exact_operations = $exactOperations
    required_preconditions = $requiredPreconditions
    excluded_from_automation = @(
        'Microsoft Defender exclusions and protection settings',
        'Windows Update policy and restart settings',
        'sleep, hibernate-idle, and lid-close settings',
        'automatic logon configuration',
        'authority_active_for_runtime and t=0 establishment data'
    )
    rationale = [ordered]@{
        fast_startup = 'HiberbootEnabled=0 ensures shutdown followed by power-on does not restore a hiberboot image that could bypass the required supervisor and startup tail verification path.'
        service = 'The accepted physical gate requires unattended Automatic service startup before interactive logon; the supervisor, not Service Recovery, classifies child exit 86.'
    }
    rollback = "Use scripts/operator/b5_windows_service_rollback.ps1 with rollback state '$($resolved.RollbackState)'; it is dry-run by default and requires its own reviewed plan digest to apply."
}
$plan = Add-PayloadSha256 $plan
$planJson = $plan | ConvertTo-Json -Depth 20
Write-Output $planJson
Write-Output "PLAN_SHA256=$($plan.payload_sha256)"

if (-not $Apply) {
    Write-Output 'DRY-RUN ONLY: no files, registry values, services, tasks, policies, or runtime state were changed.'
    exit 0
}

Assert-Administrator
if ($ExpectedPlanSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'Apply requires -ExpectedPlanSha256 from the reviewed dry-run output.'
}
if ($ExpectedPlanSha256.ToLowerInvariant() -ne $plan.payload_sha256) {
    throw 'Current privileged plan differs from the reviewed plan; refusing Apply.'
}
if (Test-Path -LiteralPath $resolved.RollbackState) {
    throw "Refusing to overwrite rollback state: $($resolved.RollbackState)"
}
if ($Action -eq 'Install') {
    if ($fastStartupBefore.read_status -ne 'RESOLVED') {
        throw 'Fast Startup before-state is UNRESOLVED; refusing privileged plan application.'
    }
    if ($serviceBefore.exists) {
        throw "Service already exists: $ServiceName"
    }
    if (Test-Path -LiteralPath $wrapper) {
        throw "Refusing to overwrite wrapper: $wrapper"
    }
    if (Test-Path -LiteralPath $configuration) {
        throw "Refusing to overwrite configuration: $configuration"
    }
}

$rollbackPacket = [ordered]@{
    schema = 'eve.b5-windows-privileged-rollback-state.v1'
    source_plan_sha256 = $plan.payload_sha256
    source_action = $Action
    service_name = $ServiceName
    deploy_directory = $deploy
    wrapper = $wrapper
    configuration = $configuration
    before = $plan.before
    expected_created_files = if ($Action -eq 'Install') {
        [ordered]@{
            wrapper_sha256 = $resolved.WinSWSha256
            configuration_sha256 = $xmlSha256
        }
    } else { $null }
    fast_startup_changed = if ($Action -eq 'Install') { $fastStartupChangeRequired } else { $false }
    automatic_rollback_permitted = $false
}
$rollbackPacket = Add-PayloadSha256 $rollbackPacket
Write-AtomicJson $resolved.RollbackState $rollbackPacket

if ($Action -eq 'Install') {
    if ($fastStartupChangeRequired) {
        & reg.exe add $FastStartupRegistryPath /v HiberbootEnabled /t REG_DWORD /d 0 /f | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Fast Startup registry update failed: $LASTEXITCODE" }
    }
    New-Item -ItemType Directory -Path $deploy -Force | Out-Null
    Copy-Item -LiteralPath $resolved.WinSWPath -Destination $wrapper
    [IO.File]::WriteAllText($configuration, $xml, [Text.UTF8Encoding]::new($false))
    & $wrapper install
    if ($LASTEXITCODE -ne 0) { throw "WinSW install failed: $LASTEXITCODE" }
    & sc.exe config $ServiceName start= auto | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sc.exe config failed: $LASTEXITCODE" }
    & sc.exe failure $ServiceName reset= 3600 actions= restart/10000 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sc.exe failure failed: $LASTEXITCODE" }
} else {
    $serviceOperation = $Action.ToLowerInvariant()
    & $wrapper $serviceOperation
    if ($LASTEXITCODE -ne 0) { throw "WinSW $Action failed: $LASTEXITCODE" }
}

Write-Output 'APPLY COMPLETED. Runtime authority remains false and t=0 was not started.'
