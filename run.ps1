# Run PhoneHome in headless Blender. Pass PhoneHome options straight through:
#   .\run.ps1 --clouds 2026-09-29 --view-lon 10 --sun-lon 80
# Finds Blender via $env:BLENDER, then `blender` on PATH.
$blender = $env:BLENDER
if (-not $blender) { $blender = (Get-Command blender -ErrorAction SilentlyContinue).Source }
if (-not $blender) {
    Write-Error 'Blender not found. Set it once with: [Environment]::SetEnvironmentVariable("BLENDER", "C:\path\to\blender.exe", "User")'
    exit 1
}
& $blender --background --factory-startup --python "$PSScriptRoot\run_phonehome.py" -- @args
exit $LASTEXITCODE
