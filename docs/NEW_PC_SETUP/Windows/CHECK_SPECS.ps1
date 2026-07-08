# ============================================================================
# CHECK NEW PC SPECS - run before the full master_feature_extraction.py run
# ============================================================================
# Tells you a safe --workers value for master_feature_extraction.py on this
# machine, and confirms the HDD's contents survived the move.
#
# USAGE:
#   cd to wherever this HDD mounted, e.g. F:\acoustic_rain_gauge_ml
#   .\docs\NEW_PC_SETUP\Windows\CHECK_SPECS.ps1
#
#   To save the report to a file (e.g. to keep alongside the HDD as a record
#   of this machine's specs) - note *> not just > , since this script's
#   output must survive redirection:
#   .\docs\NEW_PC_SETUP\Windows\CHECK_SPECS.ps1 *> docs\NEW_PC_SETUP\Windows\SPECS_REPORT_$env:COMPUTERNAME.txt
# ============================================================================

Write-Output "======================================================================"
Write-Output "  1. RAM (decides safe --workers value)"
Write-Output "======================================================================"
$mem = Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory
$totalGB = [Math]::Round($mem.TotalVisibleMemorySize / 1MB, 1)
$freeGB = [Math]::Round($mem.FreePhysicalMemory / 1MB, 1)
Write-Output "  Total RAM : $totalGB GB"
Write-Output "  Free RAM  : $freeGB GB"

# Rule of thumb measured on the original PC: ~0.5GB free RAM needed per worker
# (each worker imports numpy/scipy/librosa, ~200-400MB apiece). 18 workers
# crashed there with only ~1.9GB free ("DLL load failed... paging file too
# small") - 6 was the safe default that worked.
$safeWorkers = [Math]::Max(1, [Math]::Floor($freeGB / 0.5))
Write-Output "  -> Suggested --workers (rule of thumb, ~0.5GB free RAM per worker): $safeWorkers"
Write-Output ""

Write-Output "======================================================================"
Write-Output "  2. CPU cores (upper ceiling for --workers)"
Write-Output "======================================================================"
$cores = $env:NUMBER_OF_PROCESSORS
Write-Output "  Logical processors: $cores"
Write-Output "  Script's own cap (cpu_count()-2, before the RAM-based min(6, ...)): $($cores - 2)"
Write-Output ""

Write-Output "======================================================================"
Write-Output "  3. Drives + free space"
Write-Output "======================================================================"
Get-Volume | Where-Object { $_.DriveLetter } | Select-Object DriveLetter, FileSystemLabel,
    @{N='SizeGB';E={[Math]::Round($_.Size/1GB,1)}},
    @{N='FreeGB';E={[Math]::Round($_.SizeRemaining/1GB,1)}} | Format-Table -AutoSize
Write-Output ""

Write-Output "======================================================================"
Write-Output "  4. HDD contents check (run this FROM the HDD's project folder)"
Write-Output "======================================================================"
$driveRoot = (Get-Item $PSScriptRoot).PSDrive.Root
$checks = @(
    "$($driveRoot)arg_dataset_unzip",
    "$($driveRoot)arg_cleaned_dataset",
    "$($driveRoot)acoustic_rain_gauge_ml\src\master_feature_extraction.py"
)
foreach ($path in $checks) {
    $exists = Test-Path $path
    $mark = if ($exists) { "OK  " } else { "MISSING " }
    Write-Output "  $mark $path"
}
Write-Output ""

Write-Output "======================================================================"
Write-Output "  SUMMARY"
Write-Output "======================================================================"
Write-Output "  Recommended command:"
Write-Output "    python src\master_feature_extraction.py --limit 2000 --workers $safeWorkers"
Write-Output "  (run the smoke test first regardless - confirms real throughput on this machine"
Write-Output "   before committing to the full ~7h run at this worker count)"
Write-Output ""
