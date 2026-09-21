"""Bot pools used by the local harness."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TIER1_COMPETITORS: dict[str, str] = {
    "comp_cfr_equity_v28": str(REPO_ROOT / "bots" / "competitors" / "cfr_equity_v28"),
    "comp_bayesian_exploiter": str(
        REPO_ROOT / "bots" / "competitors" / "bayesian_exploiter"
    ),
    "comp_equity_balanced": str(REPO_ROOT / "bots" / "competitors" / "equity_balanced"),
    "comp_adaptive_hydra": str(REPO_ROOT / "bots" / "competitors" / "adaptive_hydra"),
    "comp_solver_hybrid": str(REPO_ROOT / "bots" / "competitors" / "solver_hybrid"),
    "comp_equity_table": str(REPO_ROOT / "bots" / "competitors" / "equity_table"),
    "comp_cfr_lite": str(REPO_ROOT / "bots" / "competitors" / "cfr_lite"),
}

TIER2_ARCHETYPES: dict[str, str] = {
    "adv_nit": str(REPO_ROOT / "bots" / "adversarial" / "nit.py"),
    "adv_station": str(REPO_ROOT / "bots" / "adversarial" / "station.py"),
    "adv_maniac": str(REPO_ROOT / "bots" / "adversarial" / "maniac.py"),
    "adv_solver_like": str(REPO_ROOT / "bots" / "adversarial" / "solver_like.py"),
    "adv_tournament_specialist": str(
        REPO_ROOT / "bots" / "adversarial" / "tournament_specialist.py"
    ),
    "adv_random_bot": str(REPO_ROOT / "bots" / "adversarial" / "random_bot.py"),
}

TIER3_BASELINE: dict[str, str] = {
    "strong": str(REPO_ROOT / "bots" / "strong"),
}

ALL_BOTS: dict[str, str] = {
    **TIER1_COMPETITORS,
    **TIER2_ARCHETYPES,
    **TIER3_BASELINE,
}

# Opt-in only, not in ALL_BOTS (--include-extra).
EXTRA_BOTS: dict[str, str] = {
    "mc_equity": str(REPO_ROOT / "bots" / "extra" / "mc_equity"),
    "cfr_kmeans_router": str(REPO_ROOT / "bots" / "extra" / "cfr_kmeans_router"),
    "cfr_variance_guard": str(REPO_ROOT / "bots" / "extra" / "cfr_variance_guard"),
    "cfr_opp_profiler": str(REPO_ROOT / "bots" / "extra" / "cfr_opp_profiler"),
}
