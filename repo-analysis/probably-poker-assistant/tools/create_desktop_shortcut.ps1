# Create Desktop Shortcut for Poker Assistant
# Run this script once to create the shortcut

$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $Desktop "Poker Assistant.lnk"

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "D:\AI-Workstation\Antigravity\apps\poker_assistant\project\venv\Scripts\pythonw.exe"
$Shortcut.Arguments = "launch_panel.pyw"
$Shortcut.WorkingDirectory = "D:\AI-Workstation\Antigravity\apps\poker_assistant\project"
$Shortcut.Description = "Launch Poker Assistant with Control Panel and Overlay"
$Shortcut.WindowStyle = 1
$Shortcut.Save()

Write-Host "Desktop shortcut created at: $ShortcutPath" -ForegroundColor Green
Write-Host "You can now double-click 'Poker Assistant' on your desktop to launch the app!" -ForegroundColor Cyan
