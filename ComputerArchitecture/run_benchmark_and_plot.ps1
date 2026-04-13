param(
    [string]$Source = "HW2_2.25.c",
    [string]$Exe = "HW2_2.25.exe",
    [string]$Csv = "memory_benchmark.csv",
    [string]$Log = "memory_benchmark.log",
    [switch]$SkipInstall,
    [switch]$NoOpt
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$plotScript = Join-Path $scriptDir "plot_memory_benchmark.py"
$requirements = Join-Path $scriptDir "requirements.txt"
$sourcePath = Join-Path $scriptDir $Source
$exePath = Join-Path $scriptDir $Exe
$csvPath = Join-Path $scriptDir $Csv
$logPath = Join-Path $scriptDir $Log

Write-Host "[1/4] Checking tools..."
$gccCmd = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gccCmd) {
    $msysBins = @(
        "C:\msys64\ucrt64\bin",
        "C:\msys64\mingw64\bin",
        "C:\msys64\clang64\bin"
    )
    foreach ($bin in $msysBins) {
        if (Test-Path (Join-Path $bin "gcc.exe")) {
            $env:Path = "$bin;" + $env:Path
            break
        }
    }
    $gccCmd = Get-Command gcc -ErrorAction SilentlyContinue
}
if (-not $gccCmd) {
    throw "gcc not found. Please install MinGW-w64 or MSYS2 and ensure gcc is in PATH."
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "python not found. Please install Python and add it to PATH."
}

if (-not (Test-Path $sourcePath)) {
    throw "Source file not found: $sourcePath"
}

if (-not (Test-Path $plotScript)) {
    throw "Plot script not found: $plotScript"
}

Write-Host "[2/4] Compiling benchmark..."
$optFlag = if ($NoOpt) { "-O0" } else { "-O2" }
Write-Host "Using optimization flag: $optFlag"
& gcc $sourcePath $optFlag -o $exePath
if ($LASTEXITCODE -ne 0) {
    throw "Compilation failed."
}

if ((-not $SkipInstall) -and (Test-Path $requirements)) {
    Write-Host "[3/4] Installing Python dependencies..."
    & python -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }
} else {
    Write-Host "[3/4] Skipping dependency install."
}

Write-Host "[4/4] Running benchmark and plotting..."
# Run through cmd so benchmark stderr goes to log without becoming a PowerShell error record.
$cmdLine = '"{0}" 1> "{1}" 2> "{2}"' -f $exePath, $csvPath, $logPath
cmd /c $cmdLine
if ($LASTEXITCODE -ne 0) {
    throw "Benchmark execution failed."
}

& python $plotScript $csvPath
if ($LASTEXITCODE -ne 0) {
    throw "Plot generation failed."
}

Write-Host "Done."
Write-Host "CSV:  $csvPath"
Write-Host "LOG:  $logPath"
Write-Host "PNG:  $(Join-Path $scriptDir 'memory_heatmap.png')"
Write-Host "PNG:  $(Join-Path $scriptDir 'memory_lines_by_array_size.png')"
Write-Host "PNG:  $(Join-Path $scriptDir 'memory_textbook_style.png')"
Write-Host "PNG:  $(Join-Path $scriptDir 'tlb_page_size_probe.png')"
Write-Host "PNG:  $(Join-Path $scriptDir 'tlb_entries_probe.png')"
Write-Host "PNG:  $(Join-Path $scriptDir 'tlb_associativity_probe.png')"
Write-Host "CSV:  $(Join-Path $scriptDir 'tlb_assoc_benchmark.csv')"
Write-Host "PNG:  $(Join-Path $scriptDir 'tlb_associativity_conflict.png')"
Write-Host "TXT:  $(Join-Path $scriptDir 'tlb_analysis_summary.txt')"
