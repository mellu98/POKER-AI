"""Plotting, ensemble selection, and optional strategy analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import checkpoint_iter, load_json, write_json_atomic

N_ACTIONS = 5


def bench_summary(data: dict) -> dict:
    overall = data["overall"]
    return {
        "bb100": overall["bb100"],
        "ci_lo": overall["bb100_ci95"][0],
        "ci_hi": overall["bb100_ci95"][1],
        "win": overall["win_rate"],
        "bust": data["bust_rate"],
    }


def load_existing_ensemble_result(out_dir: Path) -> dict | None:
    path = out_dir / "ensemble_result.json"
    if not path.exists():
        return None
    try:
        data = load_json(path)
    except RuntimeError:
        return None
    return data if isinstance(data, dict) and data.get("bb100") is not None else None


def build_ensemble_candidates(
    checkpoints: list[Path],
    results: list[dict],
    *,
    top_k: int,
    max_models: int,
) -> list[dict]:
    if len(checkpoints) < 2 or not results:
        return []

    ckpt_by_iter = {checkpoint_iter(ckpt): ckpt for ckpt in checkpoints}
    ranked_iters = [
        row["iter"]
        for row in sorted(results, key=lambda row: row["bb100"], reverse=True)
        if row["iter"] in ckpt_by_iter
    ]
    recent_iters = sorted(ckpt_by_iter)[-max(top_k, max_models) :]
    seen: set[tuple[int, ...]] = set()
    candidates: list[dict] = []

    def add(iters: list[int], reason: str) -> None:
        key = tuple(sorted(iters))
        if len(key) < 2 or len(key) > max_models or key in seen:
            return
        if not all(it in ckpt_by_iter for it in key):
            return
        seen.add(key)
        candidates.append({"iters": list(key), "reason": reason})

    add(sorted(ckpt_by_iter), "all_checkpoints")
    for size in range(2, min(max_models, len(recent_iters)) + 1):
        add(recent_iters[-size:], f"recent_{size}")
    greedy: list[int] = []
    for it in ranked_iters[:top_k]:
        greedy.append(it)
        add(greedy, f"top_greedy_{len(greedy)}")
    ranked_subset = sorted(ranked_iters[:top_k])
    for size in range(2, min(max_models, len(ranked_subset)) + 1):
        for start in range(0, len(ranked_subset) - size + 1):
            add(ranked_subset[start : start + size], f"top_window_{size}")
    return candidates


def plot_results(
    results: list[dict],
    version: str,
    out_dir: Path,
    ensemble_result: dict | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    iters = [row["iter"] for row in results]
    bbs = [row["bb100"] for row in results]
    if len(bbs) < 3:
        return

    x = np.array(iters, dtype=float)
    y = np.array(bbs, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    ci_lo = np.array([row.get("ci_lo", row["bb100"]) for row in results])
    ci_hi = np.array([row.get("ci_hi", row["bb100"]) for row in results])

    window = min(5, len(bbs) // 2)
    rolling_x: list[int] = []
    rolling_y: list[float] = []
    for idx in range(len(bbs)):
        start = max(0, idx - window + 1)
        rolling_x.append(iters[idx])
        rolling_y.append(float(np.mean(bbs[start : idx + 1])))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=0, color="#444", linewidth=0.8, alpha=0.4)

    trend_x = np.linspace(min(iters), max(iters) * 1.25, 50)
    trend_y = slope * trend_x + intercept
    ax.plot(
        trend_x,
        trend_y,
        "--",
        color="#2196F3",
        linewidth=1.5,
        alpha=0.35,
        label=f"Trend ({slope * 100:+.1f}/100it)",
    )

    ax.fill_between(x, ci_lo, ci_hi, alpha=0.12, color="#4CAF50", zorder=2)
    ax.errorbar(
        iters,
        bbs,
        yerr=[y - ci_lo, ci_hi - y],
        fmt="none",
        ecolor="#9E9E9E",
        elinewidth=0.8,
        capsize=3,
        zorder=3,
    )

    if rolling_x:
        ax.plot(
            rolling_x,
            rolling_y,
            "-",
            color="#FF9800",
            linewidth=2.5,
            label=f"Rolling avg ({window}-ckpt)",
            zorder=4,
        )

    ax.scatter(
        iters,
        bbs,
        c="#4CAF50",
        s=55,
        zorder=5,
        edgecolors="white",
        linewidth=0.8,
        label=f"Checkpoints (n={len(bbs)})",
    )

    if ensemble_result and ensemble_result.get("bb100") is not None:
        ax.scatter(
            [max(iters)],
            [ensemble_result["bb100"]],
            c="gold",
            s=200,
            zorder=6,
            edgecolors="black",
            linewidth=2,
            marker="*",
            label=f"Ensemble: {ensemble_result['bb100']:+.1f} bb/100",
        )

    best = max(results, key=lambda r: r["bb100"])
    ax.annotate(
        f"best: {best['bb100']:+.1f}",
        xy=(best["iter"], best["bb100"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color="#2E7D32",
        fontweight="bold",
    )

    ax.set_xlabel("CFR Iteration", fontsize=11)
    ax.set_ylabel("bb/100", fontsize=11)
    ax.set_title(
        f"Deep CFR {version.upper()} Training Curve", fontsize=13, fontweight="bold"
    )

    stats = f"Mean: {np.mean(bbs):+.1f}  Std: {np.std(bbs):.1f}  Trend: {slope * 100:+.1f}/100it"
    if ensemble_result and ensemble_result.get("bb100") is not None:
        stats += f"  Ens: {ensemble_result['bb100']:+.1f}"
    ax.set_title(stats, fontsize=9, loc="right", color="#666", fontfamily="monospace")

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_path = out_dir / f"{version}_training_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_strategies(checkpoints: list[Path]) -> list[str]:
    from bot.features import (
        LEGAL_MASK_OFFSET,
        STARTING_STACK,
        board_texture,
        detect_draws,
        load_preflop_equity,
        _evaluate_hand_strength,
    )

    def card_to_idx(card: str) -> int:
        return "cdhs".index(card[1]) * 13 + "23456789TJQKA".index(card[0])

    def layer_norm(x, gamma, beta, eps=1e-5):
        mean = x.mean()
        var = ((x - mean) ** 2).mean()
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    def load_model(path: Path) -> dict:
        data = np.load(path)
        return {key: data[key] for key in data.keys()}

    def forward(x: np.ndarray, model: dict, legal: np.ndarray) -> np.ndarray:
        h = x @ model["trunk_w0"].T + model["trunk_b0"]
        h = layer_norm(h, model["trunk_ln0_g"], model["trunk_ln0_b"])
        h = np.where(h > 0, h, 0.01 * h)
        h = h @ model["trunk_w1"].T + model["trunk_b1"]
        h = layer_norm(h, model["trunk_ln1_g"], model["trunk_ln1_b"])
        h = np.where(h > 0, h, 0.01 * h)
        v = h @ model["val_w0"].T + model["val_b0"]
        v = np.where(v > 0, v, 0.01 * v)
        v = v @ model["val_w1"].T + model["val_b1"]
        a = h @ model["adv_w0"].T + model["adv_b0"]
        a = np.where(a > 0, a, 0.01 * a)
        a = a @ model["adv_w1"].T + model["adv_b1"]
        a_masked = a * legal
        n_legal = legal.sum()
        a_mean = a_masked.sum() / max(n_legal, 1.0)
        return (v + a_masked - a_mean * legal) * legal

    def get_strategy(adv: np.ndarray, legal: np.ndarray) -> np.ndarray:
        adv = np.maximum(adv, 0.0) * legal
        total = adv.sum()
        if total > 0:
            return adv / total
        n_legal = legal.sum()
        if n_legal <= 0:
            raise RuntimeError(
                f"strategy has no legal actions: legal={legal}, adv={adv}"
            )
        return legal.astype(np.float32) / n_legal

    def make_features(hole, board, street, pos, pot, stack, n_raises, has_agg, legal):
        eq_paired, eq_suited, eq_offsuit = load_preflop_equity()
        feature_dim = LEGAL_MASK_OFFSET + N_ACTIONS
        feats = np.zeros(feature_dim, dtype=np.float32)
        legal_arr = np.array(legal, dtype=np.float32)

        c0, c1 = card_to_idx(hole[0]), card_to_idx(hole[1])
        r0, r1 = c0 % 13, c1 % 13
        s0, s1 = c0 // 13, c1 // 13
        hi, lo = max(r0, r1), min(r0, r1)
        board_idx = [card_to_idx(card) for card in board]

        if r0 == r1:
            feats[0] = eq_paired[r0]
        elif s0 == s1:
            feats[0] = eq_suited[hi, lo]
        else:
            feats[0] = eq_offsuit[hi, lo]

        if street > 0 and board:
            feats[1] = _evaluate_hand_strength(c0, c1, board_idx)

        fd, oesd, gs, outs, nfd = detect_draws([c0, c1], board_idx, street)
        feats[2] = 1.0 if fd else 0.0
        feats[3] = 1.0 if oesd else 0.0
        feats[4] = 1.0 if gs else 0.0
        feats[5] = outs / 20.0
        feats[6] = 1.0 if nfd else 0.0
        feats[7] = hi / 12.0
        feats[8] = lo / 12.0
        feats[9] = 1.0 if s0 == s1 else 0.0
        feats[10] = 1.0 if r0 == r1 else 0.0
        feats[11] = (hi - lo - 1) / 12.0 if hi != lo else 0.0
        if board_idx:
            bp, bm, bw, sp = board_texture(board_idx)
            feats[12] = 1.0 if bp else 0.0
            feats[13] = 1.0 if bm else 0.0
            board_ranks = [card % 13 for card in board_idx]
            feats[14] = sum(1 for rank in board_ranks if rank > hi) / 5.0
            feats[15] = bw
            feats[16] = 1.0 if sp else 0.0

        feats[17 + street] = 1.0
        feats[21 + pos] = 1.0
        feats[27] = pot / STARTING_STACK
        feats[28] = stack / STARTING_STACK
        feats[29] = min(stack / max(pot, 1), 10.0) / 10.0
        to_call = int(pot * 0.3) if has_agg and n_raises > 0 else 0
        feats[30] = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
        for idx in range(5):
            feats[31 + idx] = 1.0

        feats[36] = min(n_raises, 4) / 4.0
        feats[37] = float(has_agg)
        feats[40] = (STARTING_STACK - stack) / STARTING_STACK
        feats[42] = to_call / max(stack, 1)
        feats[45] = 4.0 / 6.0
        feats[LEGAL_MASK_OFFSET : LEGAL_MASK_OFFSET + N_ACTIONS] = legal_arr
        return feats, legal_arr

    scenarios = {
        "AA open BTN": (["As", "Ah"], [], 0, 0, 150, 10000, 0, 0, [0, 1, 1, 1, 0]),
        "AKs face 3bet": (["As", "Ks"], [], 0, 5, 800, 9700, 2, 1, [1, 1, 1, 1, 0]),
        "Flush draw vs bet": (
            ["Jh", "Th"],
            ["Ah", "5h", "2c"],
            1,
            2,
            600,
            9600,
            1,
            1,
            [1, 1, 1, 1, 0],
        ),
        "River nuts IP": (
            ["Ah", "Kh"],
            ["Qh", "Jh", "2c", "5d", "Th"],
            3,
            0,
            2000,
            8000,
            0,
            0,
            [0, 1, 1, 1, 1],
        ),
    }

    models = [load_model(ckpt) for ckpt in checkpoints]
    iters = [checkpoint_iter(ckpt) for ckpt in checkpoints]
    sample_idx = list(range(len(models)))
    if len(models) > 12:
        step = max(1, len(models) // 10)
        sample_idx = list(range(0, len(models), step))
        if sample_idx[-1] != len(models) - 1:
            sample_idx.append(len(models) - 1)

    lines = ["STRATEGY ANALYSIS"]
    for name, args in scenarios.items():
        feats, legal = make_features(*args)
        lines.append(f"--- {name} ---")
        for idx in sample_idx:
            adv = forward(feats.copy(), models[idx], legal)
            strat = get_strategy(adv, legal)
            spread = float(np.max(adv) - np.min(adv))
            lines.append(
                f"  iter {iters[idx]:>5d}: {' '.join(f'{value:5.1%}' for value in strat)} | spread {spread:.2f}"
            )
        adv_sum = np.zeros(N_ACTIONS, dtype=np.float64)
        for model in models:
            adv_sum += forward(feats.copy(), model, legal)
        ens_strat = get_strategy((adv_sum / len(models)).astype(np.float32), legal)
        lines.append(
            f"  ens       : {' '.join(f'{value:5.1%}' for value in ens_strat)}"
        )

    if len(models) >= 10:
        last10 = models[-10:]
        stds = []
        for _, args in scenarios.items():
            feats, legal = make_features(*args)
            strategies = np.array(
                [
                    get_strategy(forward(feats.copy(), model, legal), legal)
                    for model in last10
                ]
            )
            stds.append(strategies.std(axis=0).max())
        avg_std = float(np.mean(stds))
        label = (
            "CONVERGED"
            if avg_std < 0.10
            else "NOT YET"
            if avg_std < 0.25
            else "HIGH VARIANCE"
        )
        lines.append(
            f"Convergence (avg max-action std, last 10 ckpts): {avg_std:.3f} -> {label}"
        )
    return lines


def persist_ensemble_leaderboard(out_dir: Path, leaderboard: list[dict]) -> None:
    write_json_atomic(out_dir / "ensemble_leaderboard.json", leaderboard)
