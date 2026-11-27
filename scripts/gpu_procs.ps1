# Top 5 processes by shared GPU memory on each adapter, one line per adapter (added after a spill pause in a calibration run):
# "<luid>: <pid> <process name> <shared GB>, ..." from the per-process counters, since Windows nvidia-smi shows N/A for
# per-process memory under WDDM. Used by evict_sidecar.sh. SilentlyContinue: a process exiting mid-poll must not void it.
$r = @{}
foreach ($s in (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples) {
    if ($s.InstanceName -match '^pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+)') { $r["$($Matches[2]) $($Matches[1])"] += $s.CookedValue }
}
$r.GetEnumerator() | Where-Object { $_.Value -gt 0 } |
    ForEach-Object { $l, $p = $_.Key -split ' '; [pscustomobject]@{ L = $l; P = [int]$p; V = $_.Value } } |
    Group-Object L | ForEach-Object {
        $top = $_.Group | Sort-Object V -Descending | Select-Object -First 5 | ForEach-Object {
            $n = (Get-Process -Id $_.P -ErrorAction SilentlyContinue).ProcessName
            if (-not $n) { $n = '?' }  # powershell.exe is 5.1: no ??
            "{0} {1} {2}" -f $_.P, $n, [math]::Round($_.V / 1GB, 3).ToString([cultureinfo]::InvariantCulture)
        }
        "{0}: {1}" -f $_.Name, ($top -join ', ')
    }
