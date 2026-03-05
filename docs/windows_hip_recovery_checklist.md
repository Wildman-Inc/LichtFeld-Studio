# Windows HIP Recovery Checklist

This checklist is for the current Windows machine:

- Host: `MS-S1 MAX`
- GPU: `AMD Radeon(TM) 8060S Graphics`
- Driver observed on March 6, 2026: `32.0.23027.2005`
- Windows build observed on March 6, 2026: `26200`
- Symptom: repeated `LiveKernelEvent 141` / `amdkmdag.sys` TDR during HIP training

## Goal

Return the Windows graphics stack to a predictable baseline, then run a short local smoke test for HIP training without RDP in the loop.

## Phase 1: Reset the session

1. Stop all running `LichtFeld-Studio.exe` processes.
2. Sign out of any RDP session.
3. Reboot Windows.
4. Log in on the local console, not over RDP.

Reason:

- This machine has exposed both `Microsoft Remote Display Adapter` and the AMD adapter during the failing runs.
- HIP training validation should be done on the local desktop first.

## Phase 2: Reset scheduler and watchdog settings

1. Open an elevated PowerShell.
2. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\\scripts\\windows_gpu_recovery.ps1 -Action ApplyStableDefaults
```

3. Reboot Windows.

What `ApplyStableDefaults` does:

- Forces HAGS off by writing `HwSchMode=1`
- Removes custom TDR keys such as `TdrDelay` and `TdrDdiDelay`
- Leaves Windows to use default watchdog behavior after reboot

Note:

- The HAGS registry values used here are based on current Windows behavior in the field: `1` = off, `2` = on. Microsoft documents TDR keys, but does not publish equivalent `HwSchMode` value semantics in the same way. Treat that mapping as an implementation detail that may change in future Windows builds.

## Phase 3: Reinstall the graphics driver

Preferred order:

1. Install the OEM graphics package for `MS-S1 MAX` if the vendor provides one for this AMD iGPU.
2. If no suitable OEM package exists, use AMD's Windows package with `Driver Only` or `Minimal` install.

Recommended clean install flow:

1. Uninstall AMD Software from Apps and Features, or use AMD Cleanup Utility / Factory Reset.
2. Reboot.
3. Install the chosen driver package.
4. Reboot again.

Keep disabled for the first retry:

- Radeon overlay / metrics
- recording / instant replay
- third-party GPU overlays

## Phase 4: Pre-flight checks

From an elevated PowerShell:

```powershell
.\\scripts\\windows_gpu_recovery.ps1 -Action Status
```

Expected before the smoke test:

- Active session is local, not `RDP-*`
- `HwSchMode` shows `1`
- TDR override values are absent
- AMD display driver is present

## Phase 5: Run the smoke test

From the repo root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\\scripts\\run_truck_hip_smoke.ps1
```

The smoke test:

- refuses to run under RDP unless `-AllowRdp` is explicitly supplied
- uses a short headless HIP training run
- writes logs to a timestamped output directory under `Downloads`

Success criteria:

- no `LiveKernelEvent 141`
- no AMD timeout popup
- training reaches completion and writes `train.log`

## Phase 6: If it still fails

1. Re-run:

```powershell
.\\scripts\\windows_gpu_recovery.ps1 -Action Status
```

2. Save:

- the new `train.log`
- the smoke test `stdout.log`
- the new WER / `LiveKernelEvent 141` timestamp

3. Only then try one A/B change at a time:

- switch HAGS on and retest:

```powershell
.\\scripts\\windows_gpu_recovery.ps1 -Action EnableHags
Restart-Computer
```

- or move the workload to Linux ROCm if Windows still TDRs

## References

- Microsoft TDR registry keys: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys
- AMD OEM vs reference driver guidance: https://www.amd.com/en/resources/support-articles/faqs/GPU-56.html
- AMD cleanup / factory reset guidance: https://www.amd.com/en/resources/support-articles/faqs/GPU-601.html
