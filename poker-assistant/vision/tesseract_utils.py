"""
tesseract_utils.py — Trova il binario Tesseract in modo cross-platform.

Windows: cerca prima il bundle locale (tesseract/tesseract.exe).
macOS: cerca installazioni Homebrew comuni (/opt/homebrew, /usr/local).
Fallback: cerca `tesseract` nel PATH.
"""
import platform
import shutil
from pathlib import Path


def find_tesseract_binary() -> str:
    """Restituisce il percorso del binario Tesseract per la piattaforma corrente."""
    system = platform.system()

    # Windows: bundle locale
    if system == "Windows":
        bundled = Path(__file__).parent.parent / "tesseract" / "tesseract.exe"
        if bundled.exists():
            return str(bundled)

    # macOS: Homebrew (Apple Silicon / Intel)
    if system == "Darwin":
        for path in (
            "/opt/homebrew/bin/tesseract",
            "/usr/local/bin/tesseract",
        ):
            if Path(path).exists():
                return path

    # Fallback generico: cerca nel PATH
    from_path = shutil.which("tesseract")
    if from_path:
        return from_path

    raise RuntimeError(
        "Tesseract OCR non trovato.\n"
        "- Windows: assicurati che esista 'tesseract/tesseract.exe' nel progetto.\n"
        "- macOS: installa con 'brew install tesseract'.\n"
        "- Linux: installa con 'sudo apt install tesseract-ocr'."
    )
