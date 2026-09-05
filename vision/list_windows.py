import sys
import pygetwindow as gw

sys.stdout.reconfigure(encoding='utf-8')

print("=== FINESTRE VISIBILI ===")
for w in gw.getAllWindows():
    if w.visible and w.title.strip():
        try:
            print(f"- {w.title!r}")
        except Exception:
            print(f"- <titolo non leggibile>")
