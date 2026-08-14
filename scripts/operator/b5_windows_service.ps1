[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Install', 'Start', 'Stop', 'Restart', 'Status', 'Uninstall')]
    [string]$Action,

    [string]$ServiceName = 'EveB5Supervisor',
    [string]$DeployDirectory,
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

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'B5 Windows service operation requires an elevated administrator token.'
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
        throw "$Field is required for Install."
    }
    $resolved = [IO.Path]::GetFullPath($Value)
    if ($MustExist -and -not (Test-Path -LiteralPath $resolved)) {
        throw "$Field does not exist: $resolved"
    }
    return $resolved
}

Assert-Administrator

if ([string]::IsNullOrWhiteSpace($DeployDirectory)) {
    throw 'DeployDirectory is required.'
}
$deploy = [IO.Path]::GetFullPath($DeployDirectory)
$wrapper = Join-Path $deploy ($ServiceName + '.exe')
$configuration = Join-Path $deploy ($ServiceName + '.xml')

if ($Action -eq 'Install') {
    if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
        throw "Service already exists: $ServiceName"
    }
    $winsw = Resolve-RequiredPath $WinSWPath 'WinSWPath'
    if ($WinSWSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'WinSWSha256 must be an exact SHA-256 digest.'
    }
    $actualWinSWHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $winsw).Hash.ToLowerInvariant()
    if ($actualWinSWHash -ne $WinSWSha256.ToLowerInvariant()) {
        throw "WinSW SHA-256 differs: $actualWinSWHash"
    }
    $python = Resolve-RequiredPath $PythonPath 'PythonPath'
    $repo = Resolve-RequiredPath $RepoPath 'RepoPath'
    $store = Resolve-RequiredPath $AuthorityStore 'AuthorityStore'
    $receipt = Resolve-RequiredPath $RuntimeReceipt 'RuntimeReceipt'
    $sentinel = Resolve-RequiredPath $SentinelPath 'SentinelPath' $false
    $audit = Resolve-RequiredPath $AuditLog 'AuditLog' $false
    $alert = Resolve-RequiredPath $AlertLog 'AlertLog' $false
    $state = Resolve-RequiredPath $StateFile 'StateFile' $false
    $control = Resolve-RequiredPath $ControlFile 'ControlFile'
    $childLog = Resolve-RequiredPath $ChildRawLog 'ChildRawLog' $false
    $ready = Resolve-RequiredPath $ChildReadyFile 'ChildReadyFile' $false
    $supervisor = Resolve-RequiredPath (Join-Path $repo 'scripts\operator\b5_windows_supervisor.py') 'SupervisorScript'
    $probe = Resolve-RequiredPath (Join-Path $repo 'scripts\operator\b5_runtime_probe.py') 'RuntimeProbeScript'

    New-Item -ItemType Directory -Path $deploy -Force | Out-Null
    Copy-Item -LiteralPath $winsw -Destination $wrapper -Force
    $arguments = @(
        (Quote-Argument $supervisor), 'run',
        '--sentinel', (Quote-Argument $sentinel),
        '--audit-log', (Quote-Argument $audit),
        '--alert-log', (Quote-Argument $alert),
        '--state-file', (Quote-Argument $state),
        '--authority-store', (Quote-Argument $store),
        '--runtime-receipt', (Quote-Argument $receipt),
        '--backoff-initial', '1', '--backoff-max', '60', '--',
        (Quote-Argument $python), (Quote-Argument $probe),
        '--database', (Quote-Argument $store),
        '--control', (Quote-Argument $control),
        '--raw-log', (Quote-Argument $childLog),
        '--ready', (Quote-Argument $ready)
    ) -join ' '
    $xml = @"
<service>
  <id>$(Escape-Xml $ServiceName)</id>
  <name>$(Escape-Xml $ServiceName)</name>
  <description>EVE B5 exit-aware supervisor. The service wrapper never starts EVE directly.</description>
  <executable>$(Escape-Xml $python)</executable>
  <arguments>$arguments</arguments>
  <workingdirectory>$(Escape-Xml $repo)</workingdirectory>
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
    [IO.File]::WriteAllText($configuration, $xml, [Text.UTF8Encoding]::new($false))
    & $wrapper install
    if ($LASTEXITCODE -ne 0) { throw "WinSW install failed: $LASTEXITCODE" }
    sc.exe config $ServiceName start= auto | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sc.exe config failed: $LASTEXITCODE" }
    sc.exe failure $ServiceName reset= 3600 actions= restart/10000 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "sc.exe failure failed: $LASTEXITCODE" }
    Get-Service -Name $ServiceName | Select-Object Name, Status, StartType
    exit 0
}

if (-not (Test-Path -LiteralPath $wrapper)) {
    throw "Installed WinSW wrapper is absent: $wrapper"
}

switch ($Action) {
    'Start' {
        & $wrapper start
        if ($LASTEXITCODE -ne 0) { throw "WinSW start failed: $LASTEXITCODE" }
    }
    'Stop' {
        & $wrapper stop
        if ($LASTEXITCODE -ne 0) { throw "WinSW stop failed: $LASTEXITCODE" }
    }
    'Restart' {
        & $wrapper restart
        if ($LASTEXITCODE -ne 0) { throw "WinSW restart failed: $LASTEXITCODE" }
    }
    'Status' {
        & $wrapper status
        $wrapperStatus = $LASTEXITCODE
        sc.exe queryex $ServiceName
        sc.exe qc $ServiceName
        sc.exe qfailure $ServiceName
        exit $wrapperStatus
    }
    'Uninstall' {
        & $wrapper stopwait
        & $wrapper uninstall
        if ($LASTEXITCODE -ne 0) { throw "WinSW uninstall failed: $LASTEXITCODE" }
    }
}
