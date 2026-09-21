"""
Overlay UI — always-on-top Tkinter window showing real-time poker advice.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional


class PokerOverlay:
    """
    Simple always-on-top overlay for the poker assistant.
    """

    def __init__(self, title: str = "Poker AI Assistant"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("640x480")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        # Style
        self.root.configure(bg="#1e1e1e")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 16))
        style.configure("Header.TLabel", foreground="#00ff88", font=("Segoe UI", 22, "bold"))
        style.configure("Big.TLabel", foreground="#ffcc00", font=("Segoe UI", 36, "bold"))
        style.configure("Action.TLabel", foreground="#00ccff", font=("Segoe UI", 28, "bold"))

        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # Equity
        ttk.Label(container, text="EQUITY", style="Header.TLabel").pack(anchor=tk.W)
        self.lbl_equity = ttk.Label(container, text="--", style="Big.TLabel")
        self.lbl_equity.pack(anchor=tk.W)

        # Recommended action
        ttk.Label(container, text="ACTION", style="Header.TLabel").pack(anchor=tk.W, pady=(10, 0))
        self.lbl_action = ttk.Label(container, text="--", style="Action.TLabel")
        self.lbl_action.pack(anchor=tk.W)

        # Sizing / EV
        self.lbl_sizing = ttk.Label(container, text="")
        self.lbl_sizing.pack(anchor=tk.W)

        # Hand info
        self.lbl_hand = ttk.Label(container, text="")
        self.lbl_hand.pack(anchor=tk.W, pady=(10, 0))

        # Status / errors
        self.lbl_status = ttk.Label(container, text="Waiting for state...", foreground="#888888")
        self.lbl_status.pack(side=tk.BOTTOM, anchor=tk.W)

    def update(
        self,
        equity: Optional[float] = None,
        action: Optional[str] = None,
        sizing: Optional[str] = None,
        hand: Optional[str] = None,
        board: Optional[str] = None,
        status: Optional[str] = None,
    ):
        # Schedule all widget updates on the Tkinter main thread
        self.root.after(0, self._do_update, equity, action, sizing, hand, board, status)

    def _do_update(self, equity, action, sizing, hand, board, status):
        if equity is not None:
            self.lbl_equity.configure(text=f"{equity:.1%}")
        if action is not None:
            self.lbl_action.configure(text=action.upper())
        if sizing is not None:
            self.lbl_sizing.configure(text=f"Sizing: {sizing}")
        if hand is not None:
            b = f"  Board: {board}" if board else ""
            self.lbl_hand.configure(text=f"Hand: {hand}{b}")
        if status is not None:
            self.lbl_status.configure(text=status)

        self.root.update_idletasks()

    def run(self):
        self.root.mainloop()

    def close(self):
        # Schedule destroy on the Tkinter main thread to avoid threading issues
        try:
            self.root.after(0, self.root.destroy)
        except tk.TclError:
            pass


if __name__ == "__main__":
    overlay = PokerOverlay()
    # Simulate a few updates
    import time
    overlay.update(equity=0.62, action="RAISE", sizing="b75", hand="As Kh", board="Qd Jh 2c")
    overlay.run()
