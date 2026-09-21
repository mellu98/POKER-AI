# Changelog

All notable changes to `poker-gto-assistant` are documented here.
This project loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- README coverage for the two standalone browser front-ends (`poker_assistant.html`,
  `poker_advanced.html`) and their Node-driven JS test suites.
- `CHANGELOG.md` and `CITATION.cff`.

### Changed
- README repo-layout block now matches the actual tree (no `charts/` directory
  was ever committed; preflop ranges live as hardcoded sets in
  `poker_agent/preflop.py`).
- Test instructions split Python (`pytest`) from JS (`node`).
- Code-LOC badge replaced with a test-count badge that reflects what actually
  runs in CI.

### Fixed
- `poker_agent/ui.py`: when `to_call > 0` and the hero sat in UTG, the
  synthesized villain action was tagged with `Position.UTG` and dropped by the
  adviser's `a.player != hero_pos` filter, causing the advisor to misclassify
  the hand as RFI instead of facing-raise. The synthesized raiser is now picked
  to be some non-hero position.

### Removed
- `pandas` and `numpy` from `requirements.txt` — neither was ever imported.
- Unused `Card` import in `poker_agent/solver.py`.

## [0.1.0] — 2026-05-XX

Initial commit: Streamlit UI + Monte Carlo equity + preflop charts +
TexasSolver wrapper + YOLOv8 stub for iPhone Continuity Camera capture.
