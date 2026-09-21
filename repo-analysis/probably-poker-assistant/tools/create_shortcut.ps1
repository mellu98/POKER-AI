# Create Desktop Shortcut for Poker Assistant Control Panel
# Run this script to create a desktop shortcut

param(
    [string]$ShortcutName = "Poker Assistant",
    [switch]$StartMenu
)

$ErrorActionPreference = "Stop"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Path to the launch script
$LaunchScript = Join-Path $ProjectRoot "launch_panel.pyw"

# Check if launch script exists
if (-not (Test-Path $LaunchScript)) {
    Write-Error "Launch script not found: $LaunchScript"
    exit 1
}

# Find Python executable
$PythonPath = $null

# Try venv first
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\pythonw.exe"
if (Test-Path $VenvPython) {
    $PythonPath = $VenvPython
}
else {
    # Try system Python
    $SystemPython = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($SystemPython) {
        $PythonPath = $SystemPython.Source
    }
    else {
        # Fallback to python.exe
        $SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($SystemPython) {
            $PythonPath = $SystemPython.Source
            Write-Warning "Using python.exe instead of pythonw.exe - a console window may appear"
        }
        else {
            Write-Error "Python not found. Please install Python or create a virtual environment."
            exit 1
        }
    }
}

Write-Host "Using Python: $PythonPath"

# Icon path
$IconPath = Join-Path $ProjectRoot "assets\icons\poker_icon.ico"
if (-not (Test-Path $IconPath)) {
    Write-Warning "Icon not found at $IconPath - shortcut will use default icon"
    $IconPath = $null
}

# Determine shortcut location
if ($StartMenu) {
    $ShortcutDir = [Environment]::GetFolderPath("StartMenu")
    $ShortcutDir = Join-Path $ShortcutDir "Programs"
}
else {
    $ShortcutDir = [Environment]::GetFolderPath("Desktop")
}

$ShortcutPath = Join-Path $ShortcutDir "$ShortcutName.lnk"

# Create shortcut using WScript.Shell
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut($ShortcutPath)

$Shortcut.TargetPath = $PythonPath
$Shortcut.Arguments = "`"$LaunchScript`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Poker Assistant Control Panel"

if ($IconPath -and (Test-Path $IconPath)) {
    $Shortcut.IconLocation = $IconPath
}

$Shortcut.Save()

Write-Host ""
Write-Host "========================================"
Write-Host "Shortcut created successfully!"
Write-Host "========================================"
Write-Host ""
Write-Host "Location: $ShortcutPath"
Write-Host ""
Write-Host "Double-click the shortcut to launch Poker Assistant."
Write-Host ""

# Optionally open the folder
$OpenFolder = Read-Host "Open folder containing shortcut? (y/n)"
if ($OpenFolder -eq 'y' -or $OpenFolder -eq 'Y') {
    explorer.exe $ShortcutDir
}
