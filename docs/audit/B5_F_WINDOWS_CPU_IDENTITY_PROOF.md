# B5-F Windows CPU identity proof

Date: 2026-08-15

Status: implementation correction; the physical Windows gate remains
`UNRESOLVED` until a receipt is finalized on the exact merged head.  Runtime
authority remains false and t=0 has not started.

## Observed blocker

The Windows 11 Ryzen 7 8840U workstation completed the reboot, exit-93,
exit-86, sentinel-restart, and operator-clear observations.  Final receipt
construction then failed closed because Python reported only:

```text
AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
```

The physical-gate code required `8840U` in `platform.processor()`.  On this
Windows installation that API exposes the generic processor identifier, not
the model name.  The same host's read-only `Win32_Processor` observation
reported:

```text
AMD Ryzen 7 8840U w/ Radeon 780M Graphics
```

No environment variable was overridden and the generic identifier was not
treated as a passing model observation.

## Correction

Windows physical capture and finalization now query `Win32_Processor` through
noninteractive PowerShell.  Evidence retains the exact command, return code,
stdout, stderr, model name, manufacturer, processor ID, description, and the
separate `platform.processor()` value.  Finalization requires a successful
query, a nonempty WMI model name, Windows, and `8840U` in that model name.

A failed command, invalid JSON, absent model name, non-Windows host, or a model
without `8840U` remains fail-closed.  Linux CI cannot substitute for the
physical model observation; unit tests cover deterministic parsing and the
missing-name failure.

## Scope and evidence consequence

This correction changes only physical-evidence host identification and its
tests.  It performs no service, registry, Defender, Windows Update, power-plan,
authority-store, runtime-authority, or t=0 mutation.  Existing raw observations
remain historical inputs, but the final physical receipt and its green
preflight must be bound to the exact final head before B5 can be claimed
complete.
