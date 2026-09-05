import pygetwindow as gw
import time

print("Ridimensiona la finestra del browser trascinando l'angolo in basso a destra.")
print("Devi arrivare esattamente a: 1999 x 1249")
print("Premi Ctrl+C quando hai finito.\n")

try:
    while True:
        wins = [w for w in gw.getWindowsWithTitle("Free Poker") if w.visible]
        if wins:
            w = wins[0]
            ok = "✅ OK" if w.width == 1999 and w.height == 1249 else ""
            print(f"\rDimensione attuale: {w.width} x {w.height}   {ok}     ", end="")
        else:
            print("\rFinestra non trovata...              ", end="")
        time.sleep(0.3)
except KeyboardInterrupt:
    print("\n\nFatto!")
