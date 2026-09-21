"""
Verify all dependencies are installed correctly.

This script checks for required packages and handles platform-specific
dependencies gracefully.
"""
import sys

IS_WINDOWS = sys.platform == "win32"


def test_imports():
    """Test that all required packages can be imported."""
    # Core packages (cross-platform)
    core_tests = {
        "NumPy": lambda: __import__("numpy"),
        "Pillow": lambda: __import__("PIL"),
        "OpenCV": lambda: __import__("cv2"),
        "MSS": lambda: __import__("mss"),
        "PyTorch": lambda: __import__("torch"),
        "TorchVision": lambda: __import__("torchvision"),
        "Pytesseract": lambda: __import__("pytesseract"),
        "PyQt5": lambda: __import__("PyQt5"),
    }

    # Optional packages
    optional_tests = {
        "Ultralytics": lambda: __import__("ultralytics"),
    }

    # Windows-only packages
    windows_tests = {
        "Win32": lambda: __import__("win32gui"),
    }

    print("Testing package imports...")
    print("=" * 50)

    all_passed = True
    warnings = []

    # Test core packages
    for name, import_func in core_tests.items():
        try:
            import_func()
            print(f"[OK] {name:<15}")
        except ImportError as e:
            print(f"[FAIL] {name:<15} - {e}")
            all_passed = False

    # Test optional packages
    for name, import_func in optional_tests.items():
        try:
            import_func()
            print(f"[OK] {name:<15}")
        except ImportError as e:
            print(f"[SKIP] {name:<15} - Optional, not installed")
            warnings.append(f"{name} not installed (optional)")

    # Test Windows-only packages
    if IS_WINDOWS:
        for name, import_func in windows_tests.items():
            try:
                import_func()
                print(f"[OK] {name:<15}")
            except ImportError as e:
                print(f"[FAIL] {name:<15} - {e}")
                all_passed = False
    else:
        for name in windows_tests.keys():
            print(f"[SKIP] {name:<15} - Windows-only (not on Windows)")

    print("=" * 50)

    if all_passed:
        print("\n[SUCCESS] All packages installed successfully!")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
    else:
        print("\n[ERROR] Some packages failed to import. Please reinstall.")
        return 1

    return 0


def test_cuda():
    """Test CUDA availability for GPU acceleration."""
    print("\nTesting CUDA/GPU support...")
    print("=" * 50)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print("\n[OK] GPU acceleration ready!")
        else:
            print("\n[WARN] CUDA not available - will use CPU only")
            print("This will be slower but still functional")

    except Exception as e:
        print(f"\n[FAIL] Error testing CUDA: {e}")
        return 1

    print("=" * 50)
    return 0


def test_project_structure():
    """Verify project directory structure."""
    from pathlib import Path

    print("\nVerifying project structure...")
    print("=" * 50)

    required_dirs = [
        "src", "src/capture", "src/detection", "src/strategy", "src/ui", "src/utils",
        "config", "database", "models", "models/card_detector", "models/card_templates",
        "screenshots", "screenshots/test_hands", "screenshots/calibration",
        "logs"
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"[OK] {dir_path}")
        else:
            print(f"[FAIL] {dir_path} - MISSING")
            all_exist = False

    print("=" * 50)

    if all_exist:
        print("\n[OK] Project structure complete!")
    else:
        print("\n[FAIL] Some directories missing. Please create them.")
        return 1

    return 0


def test_platform():
    """Display platform information."""
    print("\nPlatform Information...")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Windows: {IS_WINDOWS}")

    if not IS_WINDOWS:
        print("\n[INFO] This application is designed for Windows.")
        print("       Some features (window detection) will be disabled.")

    print("=" * 50)
    return 0


if __name__ == "__main__":
    result = 0

    result |= test_platform()
    result |= test_imports()
    result |= test_cuda()
    result |= test_project_structure()

    print("\n" + "=" * 50)
    if result == 0:
        print("[OK] PHASE 1 COMPLETE - Ready for Phase 2!")
    else:
        print("[FAIL] Some tests failed - please fix before continuing")
    print("=" * 50)

    sys.exit(result)
