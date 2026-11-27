# One line per GPU adapter: "<luid> <dedicated GB> <shared GB>" (adapter-level counters). Used by watch.sh.
$r = @{}
foreach ($s in (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction Stop).CounterSamples) {
    $l = $s.InstanceName -replace '_phys.*', ''
    if (-not $r.ContainsKey($l)) { $r[$l] = @(0.0, 0.0) }
    if ($s.Path -like '*dedicated usage') { $r[$l][0] += $s.CookedValue } else { $r[$l][1] += $s.CookedValue }
}
foreach ($k in $r.Keys) {
    "{0} {1} {2}" -f $k, [math]::Round($r[$k][0] / 1GB, 3).ToString([cultureinfo]::InvariantCulture), [math]::Round($r[$k][1] / 1GB, 3).ToString([cultureinfo]::InvariantCulture)
}
