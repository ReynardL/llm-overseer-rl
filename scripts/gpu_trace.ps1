# The sidecar's 60 s trace in one Get-Counter call. It prints one line per GPU adapter,
# "<luid> <dedicated GB> <shared GB>", from the same counters scripts/gpu_mem.ps1 reads for watch.sh. It adds a "host" line
# for the host-RAM gap: process objects (exited processes the kernel still holds count here) against live processes,
# paged and nonpaged pool, and available MB.
$c = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage', '\Objects\Processes',
    '\Memory\Pool Paged Bytes', '\Memory\Pool Nonpaged Bytes', '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples
$r = @{}; $h = @{}
foreach ($s in $c) {
    if ($s.Path -like '*\gpu adapter memory(*') {
        $l = $s.InstanceName -replace '_phys.*', ''
        if (-not $r.ContainsKey($l)) { $r[$l] = @(0.0, 0.0) }
        if ($s.Path -like '*dedicated usage') { $r[$l][0] += $s.CookedValue } else { $r[$l][1] += $s.CookedValue }
    } else { $h[$s.Path.Split('\')[-1]] = $s.CookedValue }
}
$gb = { param($v) [math]::Round($v / 1GB, 3).ToString([cultureinfo]::InvariantCulture) }
foreach ($k in $r.Keys) { "{0} {1} {2}" -f $k, (& $gb $r[$k][0]), (& $gb $r[$k][1]) }
"host process_objects={0} live_processes={1} pool_paged_GB={2} pool_nonpaged_GB={3} available_MB={4}" -f `
    $h['processes'], @(Get-Process).Count, (& $gb $h['pool paged bytes']), (& $gb $h['pool nonpaged bytes']), $h['available mbytes']
