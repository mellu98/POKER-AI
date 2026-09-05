"""
Screen capture utilities — cross-platform.
Supports desktop screenshot (mss), window capture (Windows/macOS), and webcam (OpenCV).
"""
import platform
from typing import Optional

import cv2
import numpy as np
from mss import mss


def _set_dpi_aware() -> None:
    """Make Windows process DPI-aware so window coords are physical pixels."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_set_dpi_aware()


def _get_dpi_scale() -> float:
    """Return system DPI / 96.0 (Windows only)."""
    if platform.system() != "Windows":
        return 1.0
    try:
        import ctypes
        dc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(dc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, dc)
        return dpi / 96.0
    except Exception:
        return 1.0


def _get_window_rect_win(hwnd: int):
    """Return physical (left, top, right, bottom) for a Windows window handle."""
    import ctypes
    from ctypes import wintypes

    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def _find_window_rect_mac(title: str) -> Optional[dict]:
    """
    Find a window by title substring on macOS using Quartz.

    Returns dict {'left', 'top', 'width', 'height'} in mss screen coordinates,
    or None if not found / Quartz unavailable.
    """
    if platform.system() != "Darwin":
        return None
    try:
        import Quartz
    except Exception:
        print(
            "[capture] Quartz/PyObjC not installed; "
            "install with: pip install pyobjc pyobjc-framework-Quartz"
        )
        return None

    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListExcludeDesktopElements | Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
    )

    # Screen height to convert Quartz bottom-left Y to mss top-left Y
    with mss() as sct:
        screen_h = sct.monitors[0]["height"]

    for win in window_list:
        win_title = win.get(Quartz.kCGWindowName, "") or ""
        if title not in win_title:
            continue
        bounds = win.get(Quartz.kCGWindowBounds, {})
        if not bounds:
            continue
        x = int(bounds.get("X", 0))
        y = int(bounds.get("Y", 0))
        w = int(bounds.get("Width", 0))
        h = int(bounds.get("Height", 0))
        if w <= 0 or h <= 0:
            continue
        # Quartz coords: origin at bottom-left; mss uses top-left
        top = screen_h - (y + h)
        return {"left": x, "top": top, "width": w, "height": h}

    return None


def _capture_with_mss(monitor: dict) -> np.ndarray:
    """Capture a region using mss and return BGR numpy array."""
    with mss() as sct:
        img = sct.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)


def screenshot(monitor: Optional[dict] = None, window_title: Optional[str] = None) -> np.ndarray:
    """
    Capture the screen and return it as a BGR numpy array.

    Args:
        monitor: dict with keys 'top', 'left', 'width', 'height'.
                 If None, captures the primary monitor.
        window_title: If provided, captures the first visible window whose title
                      contains this substring (e.g. "Poker - Opera").
    """
    system = platform.system()

    if window_title:
        # macOS path
        if system == "Darwin":
            rect = _find_window_rect_mac(window_title)
            if rect:
                print(f"[capture] Using macOS window: {window_title!r} {rect}")
                return _capture_with_mss(rect)

        # Windows path
        if system == "Windows":
            try:
                import pygetwindow as gw

                windows = [w for w in gw.getWindowsWithTitle(window_title) if w.visible]
                if windows:
                    w = windows[0]
                    left, top, right, bottom = _get_window_rect_win(w._hWnd)
                    phys_w = right - left
                    phys_h = bottom - top
                    scale = _get_dpi_scale()
                    log_w = int(phys_w / scale)
                    log_h = int(phys_h / scale)
                    print(
                        f"[capture] Using window: {w.title!r} "
                        f"physical=({phys_w}x{phys_h}) logical=({log_w}x{log_h}) scale={scale:.2f}"
                    )
                    from PIL import ImageGrab

                    bbox = (left, top, right, bottom)
                    img = ImageGrab.grab(bbox=bbox)
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    if frame.shape[1] != log_w or frame.shape[0] != log_h:
                        frame = cv2.resize(
                            frame, (log_w, log_h), interpolation=cv2.INTER_LANCZOS4
                        )
                    return frame
            except Exception as e:
                print(f"[capture] Window capture failed: {e}")

        print(f"[capture] Window '{window_title}' not found; falling back to full screen")

    if monitor is None:
        with mss() as sct:
            monitor = sct.monitors[1]
    return _capture_with_mss(monitor)


def webcam_capture(device: int = 0) -> Optional[np.ndarray]:
    """Capture a single frame from the webcam."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def crop_roi(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop a region of interest from the frame."""
    return frame[y : y + h, x : x + w]
