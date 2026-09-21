# Poker AI Assistant

An AI-powered poker assistant that uses computer vision to watch your poker game in real-time and provides strategic suggestions via a transparent overlay. Currently configured for Ignition Casino.

## 🎯 Purpose

This is a **learning and practice tool** designed for use with play money only on PokerStars simulator. It demonstrates advanced concepts in:

- Computer Vision & Image Processing
- Machine Learning & AI
- Real-time Systems
- Game Theory & Probability
- Software Architecture

## ⚠️ Disclaimer

**For Educational Purposes Only**: This tool is designed for learning poker strategy and practicing with play money. Using automated assistance in real-money poker games may violate terms of service and could result in account penalties. Use responsibly.

## 🛠️ System Requirements

- **OS**: Windows 11
- **Python**: 3.11+ (tested with 3.13.5)
- **GPU**: RTX 4060 8GB or similar (optional, for GPU acceleration)
- **RAM**: 64GB recommended
- **Software**: Poker client (configured for Ignition Casino)

## 📋 Features

### Phase 1: Environment Setup ✅

- [x] Project structure and virtual environment
- [x] Dependency management
- [x] Configuration system
- [x] Logging infrastructure

### Phase 2: Screen Capture ✅

- [x] Window detection for PokerStars
- [x] Region-based screen capture
- [x] Interactive calibration tool
- [x] Screenshot management

### Phase 3: Card Detection ✅

- [x] Template matching for card recognition
- [x] OCR for chip counts and pot amounts
- [x] Game state tracking
- [x] Detection accuracy optimization

### Phase 4: Strategy Engine ✅

- [x] Hand evaluation (all poker hands)
- [x] Equity calculator (Monte Carlo simulation)
- [x] Pot odds calculator
- [x] Decision engine with strategic recommendations

### Phase 5: Overlay & Integration ✅

- [x] Transparent overlay window
- [x] Real-time display updates
- [x] Complete system integration
- [x] Performance optimization

## 🚀 Quick Start

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ProbablyMaybeNo/poker-assistant.git
   cd poker-assistant
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**

   ```bash
   python test_installation.py
   ```

### Running the Application

```bash
# Activate virtual environment
venv\Scripts\activate

# Run main application
python src/main.py
```

## 📁 Project Structure

```
poker_assistant/
├── config/              # Configuration files
│   ├── settings.json    # Application settings
│   ├── regions.json     # Screen region coordinates
│   └── poker_rules.json # Game constants
├── database/            # Strategy databases and charts
├── logs/                # Application logs
├── models/              # AI models and card templates
│   ├── card_detector/   # YOLO model files
│   └── card_templates/  # Template matching images
├── screenshots/         # Test images and calibration
├── src/                 # Source code
│   ├── capture/         # Screen capture system
│   ├── detection/       # Card and text detection
│   ├── strategy/        # Hand evaluation and decision engine
│   ├── ui/              # Overlay and calibration tools
│   └── utils/           # Logger and config utilities
├── tools/               # Utility scripts
├── venv/                # Virtual environment
├── .gitignore
├── README.md
├── requirements.txt
└── test_installation.py
```

## 🧪 Testing

Run the installation verification:

```bash
python test_installation.py
```

This will verify:

- All required packages are installed
- CUDA/GPU availability (optional)
- Project directory structure
- Configuration files

## 📚 Dependencies

### Core Libraries

- **numpy** - Numerical computing
- **pillow** - Image processing
- **opencv-python** - Computer vision
- **mss** - Screen capture

### AI/ML Libraries

- **torch** - PyTorch deep learning framework
- **torchvision** - Computer vision models
- **ultralytics** - YOLO object detection

### OCR & UI

- **pytesseract** - Optical character recognition
- **PyQt5** - GUI framework

### Utilities

- **python-dotenv** - Environment configuration
- **tqdm** - Progress bars
- **pywin32** - Windows API access

See `requirements.txt` for complete list with versions.

## 🎓 Learning Outcomes

By building and using this project, you'll gain experience with:

- **Computer Vision**: Template matching, image preprocessing, OCR, object detection
- **Machine Learning**: Monte Carlo simulation, probability calculations, decision trees
- **Software Engineering**: Modular architecture, configuration management, logging
- **Game Theory**: Poker hand rankings, pot odds, equity, position-based strategy
- **Python Development**: PyTorch, OpenCV, PyQt5, multi-threaded applications

## 🔧 Configuration

### Settings (`config/settings.json`)

Configure capture interval, detection thresholds, overlay appearance, strategy style, and performance parameters.

### Regions (`config/regions.json`)

Define screen regions for cards, pot, player stack, and action buttons. Use the calibration tool (Phase 2) to set these coordinates.

## 🐛 Troubleshooting

### Common Issues

**Python not found**

- Install Python 3.11+ and add to PATH

**Virtual environment activation fails**

- Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell

**PyTorch CUDA not available**

- Update NVIDIA drivers
- Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

**PokerStars window not found**

- Ensure PokerStars is running
- Check window title in `config/settings.json`
- Try running as administrator

**Cards not detected**

- Recalibrate regions using the calibration tool
- Adjust confidence threshold in settings
- Verify screenshot quality

See `logs/errors.log` for detailed error information.

## 📝 Development Status

**Current Version**: V1.0 Release Candidate ✅
**Status**: Feature complete - ready for testing

### Completion Summary

- [x] Phase 1: Environment Setup - Complete
- [x] Phase 2: Screen Capture System - Complete
- [x] Phase 3: Card Detection - Complete
- [x] Phase 4: Strategy Engine - Complete
- [x] Phase 5: Overlay & Integration - Complete

### V1 Enhancements (All Complete) ✅
- [x] GTO preflop ranges (open/3bet/4bet for all positions)
- [x] Postflop strategy with board texture analysis
- [x] Enhanced overlay with pot odds and EV display
- [x] Session logging for learning data
- [x] Performance monitoring and bottleneck detection

### V2 Roadmap
- [ ] Multi-table support
- [ ] Opponent modeling (VPIP/PFR/AF stats)
- [ ] Hand history export
- [ ] GUI calibration tool improvements

## 🤝 Contributing

This is an educational project. Contributions, suggestions, and improvements are welcome!

## 📄 License

This project is for educational purposes. Please use responsibly and in accordance with PokerStars Terms of Service.

## 🔗 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyQt5 Documentation](https://doc.qt.io/qtforpython/)
- [Poker Strategy Resources](https://upswingpoker.com/)

## 📧 Contact

**GitHub**: [@ProbablyMaybeNo](https://github.com/ProbablyMaybeNo)

---

**Remember**: This tool is for learning only. Use responsibly with play money for practice purposes. 🎯♠️♥️♣️♦️
