"""Deep CFR training loop."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import AverageStrategyNet, DuelingAdvantageNet
from .buffer import ReservoirBuffer, GPUBuffer
from .traverse_fast import run_traversals_fast
from bot.features import FEATURE_DIM, LEGAL_MASK_OFFSET, load_preflop_equity

_USE_NUMBA = False
_NUMBA_LUTS = None
_EQUITY_TABLES = None

N_ACTIONS = 5
N_PLAYERS = 6
TRUNK_DIM = 256
HEAD_DIM = 128


def _extract_dueling_weights(
    net: DuelingAdvantageNet,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for module in net.trunk:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().copy())
            biases.append(module.bias.detach().cpu().numpy().copy())
        elif isinstance(module, nn.LayerNorm):
            weights.append(module.weight.detach().cpu().numpy().copy())
            biases.append(module.bias.detach().cpu().numpy().copy())

    for module in net.value_head:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().copy())
            biases.append(module.bias.detach().cpu().numpy().copy())

    for module in net.advantage_head:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().copy())
            biases.append(module.bias.detach().cpu().numpy().copy())

    return weights, biases


def _export_strategy_npz(net: AverageStrategyNet, path: Path) -> None:
    modules = [m for m in net.trunk if isinstance(m, (nn.Linear, nn.LayerNorm))]
    arrays = {
        "strategy_w0": modules[0].weight.detach().cpu().numpy().copy(),
        "strategy_b0": modules[0].bias.detach().cpu().numpy().copy(),
        "strategy_ln0_g": modules[1].weight.detach().cpu().numpy().copy(),
        "strategy_ln0_b": modules[1].bias.detach().cpu().numpy().copy(),
        "strategy_w1": modules[2].weight.detach().cpu().numpy().copy(),
        "strategy_b1": modules[2].bias.detach().cpu().numpy().copy(),
        "strategy_ln1_g": modules[3].weight.detach().cpu().numpy().copy(),
        "strategy_ln1_b": modules[3].bias.detach().cpu().numpy().copy(),
        "strategy_out_w": net.policy_head.weight.detach().cpu().numpy().copy(),
        "strategy_out_b": net.policy_head.bias.detach().cpu().numpy().copy(),
    }
    np.savez(str(path), **arrays)


def _export_advantage_npz(net: DuelingAdvantageNet, path: Path) -> None:
    weights, biases = _extract_dueling_weights(net)
    arrays = {
        "trunk_w0": weights[0],
        "trunk_b0": biases[0],
        "trunk_ln0_g": weights[1],
        "trunk_ln0_b": biases[1],
        "trunk_w1": weights[2],
        "trunk_b1": biases[2],
        "trunk_ln1_g": weights[3],
        "trunk_ln1_b": biases[3],
        "val_w0": weights[4],
        "val_b0": biases[4],
        "val_w1": weights[5],
        "val_b1": biases[5],
        "adv_w0": weights[6],
        "adv_b0": biases[6],
        "adv_w1": weights[7],
        "adv_b1": biases[7],
    }
    np.savez(str(path), **arrays)


def _parse_strategy_variants(spec: str) -> list[tuple[str, int, int]]:
    """Parse "name:steps:trunk,name:steps:trunk" into [(name, steps, trunk_dim)]."""
    variants: list[tuple[str, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, steps, trunk = part.split(":")
        variants.append((name.strip(), int(steps), int(trunk)))
    if not variants:
        raise ValueError(f"no strategy variants parsed from {spec!r}")
    return variants


def _dummy_arrays():
    shapes = [
        (TRUNK_DIM, FEATURE_DIM),
        TRUNK_DIM,
        TRUNK_DIM,
        TRUNK_DIM,
        (TRUNK_DIM, TRUNK_DIM),
        TRUNK_DIM,
        TRUNK_DIM,
        TRUNK_DIM,
        (HEAD_DIM, TRUNK_DIM),
        HEAD_DIM,
        (1, HEAD_DIM),
        1,
        (HEAD_DIM, TRUNK_DIM),
        HEAD_DIM,
        (N_ACTIONS, HEAD_DIM),
        N_ACTIONS,
    ]
    ones_idx = {2, 6}
    return [
        np.ones(s, dtype=np.float32) if i in ones_idx else np.zeros(s, dtype=np.float32)
        for i, s in enumerate(shapes)
    ]


def _worker_counts(total: int, n_workers: int) -> list[int]:
    per_worker = total // n_workers
    remainder = total % n_workers
    return [per_worker + (1 if idx < remainder else 0) for idx in range(n_workers)]


def _exploration_epsilon(
    iteration: int,
    n_iterations: int,
    start: float,
    end: float,
    decay_iters: int,
) -> float:
    if start <= 0.0 and end <= 0.0:
        return 0.0
    horizon = decay_iters if decay_iters > 0 else n_iterations
    if horizon <= 1:
        return max(start, 0.0)
    progress = min(max((iteration - 1) / (horizon - 1), 0.0), 1.0)
    return max(start + (end - start) * progress, 0.0)


def _collect_worker_batches(
    results: list[tuple[np.ndarray, ...]],
    buffer: ReservoirBuffer,
    strategy_buffer: ReservoirBuffer,
    iteration: int,
) -> tuple[int, int]:
    n_samples = 0
    n_strategy_samples = 0
    for result in results:
        if len(result) == 2:
            feats, advs = result
            strat_feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
            strategies = np.zeros((0, N_ACTIONS), dtype=np.float32)
            strat_weights = np.zeros((0,), dtype=np.float32)
        elif len(result) == 4:
            feats, advs, strat_feats, strategies = result
            strat_weights = np.ones((strat_feats.shape[0],), dtype=np.float32)
        else:
            feats, advs, strat_feats, strategies, strat_weights = result
        if feats.shape[0] > 0:
            buffer.add_batch(feats, advs, iteration)
            n_samples += feats.shape[0]
        if strat_feats.shape[0] > 0:
            keep = strat_weights > np.float32(1e-8)
            strat_feats = strat_feats[keep]
            strategies = strategies[keep]
            strat_weights = strat_weights[keep]
        if strat_feats.shape[0] > 0:
            strategy_buffer.add_batch(
                strat_feats,
                strategies,
                iteration,
                sample_weights=strat_weights,
            )
            n_strategy_samples += strat_feats.shape[0]
    return n_samples, n_strategy_samples


def _run_parallel_traversals(
    *,
    iteration: int,
    n_traversals: int,
    traverser: int,
    n_workers: int,
    trav_weights: list[np.ndarray],
    trav_biases: list[np.ndarray],
    opponent_exploration: float,
    randomize_stacks: bool,
) -> list[tuple[np.ndarray, ...]]:
    import multiprocessing as mp

    counts = _worker_counts(n_traversals, n_workers)
    ctx = mp.get_context("fork")

    if _USE_NUMBA:
        flush_lut, nf_keys, nf_vals = _NUMBA_LUTS
        eq_p, eq_s, eq_o = _EQUITY_TABLES
        args_list = [
            {
                "seed": iteration * 1000 + worker_idx,
                "n_traversals": count,
                "traverser": traverser,
                "n_players": N_PLAYERS,
                "weights": trav_weights,
                "biases": trav_biases,
                "flush_lut": flush_lut,
                "nf_keys": nf_keys,
                "nf_vals": nf_vals,
                "eq_paired": eq_p,
                "eq_suited": eq_s,
                "eq_offsuit": eq_o,
                "opponent_exploration": opponent_exploration,
                "randomize_stacks": randomize_stacks,
            }
            for worker_idx, count in enumerate(counts)
        ]
        from .traverse_numba import worker_run_jit

        with ctx.Pool(n_workers) as pool:
            return pool.map(worker_run_jit, args_list)

    args_list = [
        {
            "worker_id": iteration * 1000 + worker_idx,
            "n_traversals": count,
            "traverser": traverser,
            "n_players": N_PLAYERS,
            "weights": trav_weights,
            "biases": trav_biases,
            "equity_tables": _EQUITY_TABLES,
            "opponent_exploration": opponent_exploration,
            "randomize_stacks": randomize_stacks,
        }
        for worker_idx, count in enumerate(counts)
    ]
    from .worker import worker_run

    with ctx.Pool(n_workers) as pool:
        return pool.map(worker_run, args_list)


def _init_numba() -> bool:
    """Load Numba traversal + hand-eval LUTs or fail immediately."""
    global _USE_NUMBA, _NUMBA_LUTS, _EQUITY_TABLES
    from .traverse_numba import run_traversals
    from ._hand_eval_lut import load_or_generate

    lut_path = Path(__file__).resolve().parent.parent / "data" / "hand_eval_lut.npz"
    flush_lut, nf_keys, nf_vals = load_or_generate(lut_path)
    _NUMBA_LUTS = (flush_lut, nf_keys, nf_vals)

    eq_paired, eq_suited, eq_offsuit = load_preflop_equity()
    _EQUITY_TABLES = (eq_paired, eq_suited, eq_offsuit)

    w = _dummy_arrays()
    run_traversals(
        2,
        6,
        0,
        *w,
        0,
        flush_lut,
        nf_keys,
        nf_vals,
        eq_paired,
        eq_suited,
        eq_offsuit,
        0.0,
        False,
    )
    _USE_NUMBA = True
    print("[Deep CFR] Numba JIT traversal: ENABLED")
    return True


def train_network(
    net: DuelingAdvantageNet,
    buffers: list[ReservoirBuffer],
    optimizer: optim.Adam,
    n_steps: int = 4000,
    batch_size: int = 32768,
    device: torch.device = torch.device("cpu"),
    gpu_buffer: GPUBuffer | None = None,
) -> float:
    """Train advantage network on buffer samples. Returns average loss."""
    net.train()
    criterion = nn.MSELoss(reduction="none")
    total_loss = 0.0

    use_gpu = gpu_buffer is not None and len(gpu_buffer) >= batch_size * 2

    if not use_gpu:
        buf_sizes = np.array([len(b) for b in buffers], dtype=np.float64)
        buf_total = buf_sizes.sum()
        buf_probs = buf_sizes / buf_total if buf_total > 0 else None

    for step_i in range(n_steps):
        if use_gpu:
            feat_t, adv_t, w_t = gpu_buffer.sample(batch_size)
            w_t = w_t.unsqueeze(1)
        else:
            buf_idx = np.random.choice(len(buffers), p=buf_probs)
            features, advantages, weights = buffers[buf_idx].sample(batch_size)
            feat_t = torch.from_numpy(features).to(device)
            adv_t = torch.from_numpy(advantages).to(device)
            w_t = torch.from_numpy(weights).to(device).unsqueeze(1)

        legal_mask = feat_t[:, LEGAL_MASK_OFFSET : LEGAL_MASK_OFFSET + N_ACTIONS]

        # Per-sample advantage normalization across legal actions only
        with torch.no_grad():
            n_legal = legal_mask.sum(dim=1, keepdim=True).clamp(min=1)
            legal_adv = adv_t * legal_mask
            mean = legal_adv.sum(dim=1, keepdim=True) / n_legal
            var = ((legal_adv - mean * legal_mask) ** 2 * legal_mask).sum(
                dim=1, keepdim=True
            ) / n_legal
            std = (var + 1e-6).sqrt()
            adv_t = ((adv_t - mean) / std) * legal_mask

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            pred = net(feat_t, legal_mask)
            loss_per_sample = criterion(pred, adv_t).mean(dim=1, keepdim=True)
            loss = (loss_per_sample * w_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    net.eval()
    return total_loss / n_steps


def train_strategy_network(
    net: AverageStrategyNet,
    strategy_buffer: ReservoirBuffer,
    optimizer: optim.Adam,
    n_steps: int = 3000,
    batch_size: int = 8192,
    device: torch.device = torch.device("cpu"),
    milestones: list[int] | None = None,
    tag: str = "",
) -> float:
    """Train the average strategy net on stored policy targets.

    With ``milestones``, prints mean CE per step interval; a loss still falling across
    intervals means the net is undertrained at ``n_steps``.
    """
    net.train()
    total_loss = 0.0

    ms = sorted(m for m in (milestones or []) if 0 < m <= n_steps)
    ms_idx = 0
    interval_loss = 0.0
    interval_count = 0
    interval_marks: list[tuple[int, float]] = []

    for step_i in range(n_steps):
        features, targets, weights = strategy_buffer.sample(batch_size)
        feat_t = torch.from_numpy(features).to(device)
        target_t = torch.from_numpy(targets).to(device)
        w_t = torch.from_numpy(weights).to(device)
        legal_mask = feat_t[:, LEGAL_MASK_OFFSET : LEGAL_MASK_OFFSET + N_ACTIONS]

        target_t = target_t * legal_mask
        target_sum = target_t.sum(dim=1, keepdim=True).clamp(min=1e-12)
        target_t = target_t / target_sum

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            logits = net(feat_t, legal_mask)
            log_probs = torch.log_softmax(logits, dim=1)
            loss_per_sample = -(target_t * log_probs).sum(dim=1)
            loss = (loss_per_sample * w_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        loss_value = loss.item()
        total_loss += loss_value

        if ms:
            interval_loss += loss_value
            interval_count += 1
            if ms_idx < len(ms) and (step_i + 1) == ms[ms_idx]:
                interval_marks.append(
                    (ms[ms_idx], interval_loss / max(interval_count, 1))
                )
                interval_loss = 0.0
                interval_count = 0
                ms_idx += 1

    if interval_marks:
        marks = " | ".join(f"<={m}:{v:.4f}" for m, v in interval_marks)
        print(f"  [strat-probe {tag}] interval-mean CE  {marks}")

    net.eval()
    return total_loss / n_steps


def train_deep_cfr(
    n_iterations: int = 250,
    n_traversals: int = 100_000,
    buffer_capacity: int = 5_000_000,
    train_steps: int = 10_000,
    strategy_train_steps: int = 3000,
    batch_size: int = 8192,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints/deep_cfr",
    checkpoint_every: int = 10,
    device_str: str = "auto",
    seed: int = 42,
    n_workers: int = 16,
    strategy_buffer_capacity: int = 5_000_000,
    opponent_exploration_start: float = 0.10,
    opponent_exploration_end: float = 0.02,
    opponent_exploration_decay_iters: int = 0,
    randomize_stacks: bool = True,
    resume: bool = False,
    save_resume_buffers: bool = False,
    strategy_variants: list[tuple[str, int, int]] | None = None,
    export_advantage_npz: bool = False,
    strategy_milestone_every: int = 0,
) -> None:
    if strategy_variants is None:
        variants = [(None, strategy_train_steps, TRUNK_DIM)]
    else:
        variants = strategy_variants
    control_name = variants[0][0]
    strategy_milestones = [500, 1000, 2000, 3000, 5000, 8000]
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    total_buf = buffer_capacity * N_PLAYERS
    epochs = (train_steps * batch_size) / total_buf if total_buf > 0 else 0
    print(f"[Deep CFR] Device: {device}")
    print(f"[Deep CFR] Iterations: {n_iterations}, Traversals/iter: {n_traversals}")
    print(
        f"[Deep CFR] Per-player buffer: {buffer_capacity:,} x {N_PLAYERS} = {total_buf:,}"
    )
    print(f"[Deep CFR] Strategy buffer: {strategy_buffer_capacity:,} total samples")
    print(f"[Deep CFR] Epochs per training phase: {epochs:.1f}")
    print(
        f"[Deep CFR] Advantage network: Dueling (trunk=2x{TRUNK_DIM}, heads={HEAD_DIM})"
    )
    print(f"[Deep CFR] Average strategy network: trunk=2x{TRUNK_DIM}, softmax head")
    print(
        "[Deep CFR] Strategy variants: "
        + ", ".join(
            f"{name or 'default'}({steps} steps, 2x{trunk})"
            for name, steps, trunk in variants
        )
        + f" | control={control_name or 'default'} | export_advantage_npz={export_advantage_npz}"
    )
    print(f"[Deep CFR] Features: {FEATURE_DIM}, Actions: {N_ACTIONS}")
    print(f"[Deep CFR] Workers: {n_workers}")
    print(
        "[Deep CFR] Opponent exploration: "
        f"{opponent_exploration_start:.3f} -> {opponent_exploration_end:.3f} "
        f"over {opponent_exploration_decay_iters or n_iterations} iters"
    )
    print(
        f"[Deep CFR] Stack randomization: {'ENABLED' if randomize_stacks else 'disabled'}"
    )

    _init_numba()

    global _EQUITY_TABLES
    if _EQUITY_TABLES is None:
        _EQUITY_TABLES = load_preflop_equity()

    rng = np.random.default_rng(seed)

    buffers = [
        ReservoirBuffer(buffer_capacity, FEATURE_DIM, N_ACTIONS)
        for _ in range(N_PLAYERS)
    ]
    strategy_buffer = ReservoirBuffer(
        strategy_buffer_capacity,
        FEATURE_DIM,
        N_ACTIONS,
        store_sample_weights=True,
    )
    buf_vram = total_buf * (FEATURE_DIM + N_ACTIONS + 1) * 4
    gpu_buf_limit = 18e9
    if device.type == "cuda" and buf_vram <= gpu_buf_limit:
        gpu_buffer = GPUBuffer(buffers, device)
    else:
        gpu_buffer = None
        if device.type == "cuda" and buf_vram > gpu_buf_limit:
            print(
                f"[Deep CFR] Buffer {buf_vram / 1e9:.1f} GB > {gpu_buf_limit / 1e9:.0f} GB VRAM limit, using CPU sampling"
            )

    net = DuelingAdvantageNet(FEATURE_DIM, TRUNK_DIM, HEAD_DIM)
    strategy_net = AverageStrategyNet(FEATURE_DIM, TRUNK_DIM, N_ACTIONS)
    net.eval()
    strategy_net.eval()

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    resume_iter = 0
    resume_file = ckpt_path / "latest.pt"
    if resume and resume_file.exists():
        print(f"[Deep CFR] Resuming from {resume_file}")
        checkpoint = torch.load(resume_file, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["net"])
        if "strategy_net" in checkpoint:
            strategy_net.load_state_dict(checkpoint["strategy_net"])
        resume_iter = checkpoint["iteration"]
        if "buffers" in checkpoint:
            for buf, state_dict in zip(buffers, checkpoint["buffers"]):
                buf.load_state_dict(state_dict)
            if "strategy_buffer" in checkpoint:
                strategy_buffer.load_state_dict(checkpoint["strategy_buffer"])
            else:
                raise RuntimeError(
                    "Checkpoint has advantage buffers but no strategy_buffer; "
                    "resuming would discard accumulated strategy samples."
                )
        else:
            raise RuntimeError(
                "Refusing to resume without replay buffers because that changes "
                "the training distribution."
            )
    elif resume and not resume_file.exists():
        print(
            f"[Deep CFR] --resume requested but {resume_file} does not exist; starting from scratch"
        )
    elif resume_file.exists():
        print(
            f"[Deep CFR] Found {resume_file}; starting fresh because --resume was not passed."
        )

    start_time = time.time()
    avg_loss = 0.0
    avg_strategy_loss = 0.0
    current_lr = lr

    for iteration in range(resume_iter + 1, n_iterations + 1):
        iter_start = time.time()
        traverser = (iteration - 1) % N_PLAYERS
        opponent_exploration = _exploration_epsilon(
            iteration,
            n_iterations,
            opponent_exploration_start,
            opponent_exploration_end,
            opponent_exploration_decay_iters,
        )

        trav_weights, trav_biases = _extract_dueling_weights(net)
        if n_workers > 1:
            results = _run_parallel_traversals(
                iteration=iteration,
                n_traversals=n_traversals,
                traverser=traverser,
                n_workers=n_workers,
                trav_weights=trav_weights,
                trav_biases=trav_biases,
                opponent_exploration=opponent_exploration,
                randomize_stacks=randomize_stacks,
            )
            n_samples, n_strategy_samples = _collect_worker_batches(
                results, buffers[traverser], strategy_buffer, iteration
            )
        else:
            size_before = len(buffers[traverser])
            strategy_size_before = len(strategy_buffer)
            run_traversals_fast(
                n_traversals,
                N_PLAYERS,
                traverser,
                net,
                buffers[traverser],
                iteration,
                rng,
                _EQUITY_TABLES,
                opponent_exploration=opponent_exploration,
                strategy_buffer=strategy_buffer,
                randomize_stacks=randomize_stacks,
            )
            n_samples = len(buffers[traverser]) - size_before
            n_strategy_samples = len(strategy_buffer) - strategy_size_before

        total_buf_size = sum(len(b) for b in buffers)
        if total_buf_size >= batch_size * 2:
            net = DuelingAdvantageNet(FEATURE_DIM, TRUNK_DIM, HEAD_DIM)
            if iteration <= 20:
                current_lr = 1e-4 + (lr - 1e-4) * (iteration / 20)
            else:
                current_lr = lr
            optimizer = optim.Adam(net.parameters(), lr=current_lr)
            net.to(device)
            if gpu_buffer is not None:
                gpu_buffer.sync()
            avg_loss = train_network(
                net, buffers, optimizer, train_steps, batch_size, device, gpu_buffer
            )
            net.to("cpu")
            net.eval()

        strategy_nets: dict[str | None, AverageStrategyNet] = {}
        if len(strategy_buffer) >= batch_size * 2:
            probe_this_iter = (
                strategy_milestone_every > 0
                and iteration % strategy_milestone_every == 0
            )
            max_steps_name = max(variants, key=lambda v: v[1])[0]
            for vname, vsteps, vtrunk in variants:
                snet = AverageStrategyNet(FEATURE_DIM, vtrunk, N_ACTIONS)
                sopt = optim.Adam(snet.parameters(), lr=current_lr)
                snet.to(device)
                ms = (
                    strategy_milestones
                    if (probe_this_iter and vname == max_steps_name)
                    else None
                )
                vloss = train_strategy_network(
                    snet,
                    strategy_buffer,
                    sopt,
                    vsteps,
                    batch_size,
                    device,
                    milestones=ms,
                    tag=f"{vname or 'default'} iter{iteration}",
                )
                snet.to("cpu")
                snet.eval()
                strategy_nets[vname] = snet
                if vname == control_name:
                    avg_strategy_loss = vloss
            strategy_net = strategy_nets[control_name]

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        samples_per_sec = n_traversals / iter_time if iter_time > 0 else 0
        iters_done = iteration - resume_iter
        iters_left = n_iterations - iteration
        eta_sec = (elapsed / iters_done) * iters_left if iters_done > 0 else 0
        eta_min = eta_sec / 60

        buf_sizes = [len(b) for b in buffers]
        print(
            f"[Iter {iteration:4d}/{n_iterations}] "
            f"traverser={traverser} adv_samples={n_samples:,} strat_samples={n_strategy_samples:,} "
            f"adv_loss={avg_loss:.4f} strat_loss={avg_strategy_loss:.4f} "
            f"lr={current_lr:.1e} eps={opponent_exploration:.3f} "
            f"buffer={total_buf_size:,} strategy_buffer={len(strategy_buffer):,} "
            f"[{','.join(f'{s // 1000}k' for s in buf_sizes)}] "
            f"speed={samples_per_sec:.0f} trav/s "
            f"time={iter_time:.1f}s elapsed={elapsed:.0f}s "
            f"ETA={eta_min:.0f}min"
        )

        if iteration % checkpoint_every == 0 or iteration == n_iterations:
            save_path = ckpt_path / f"iter_{iteration:05d}.pt"
            ckpt_data = {
                "net": net.state_dict(),
                "strategy_net": strategy_net.state_dict(),
                "iteration": iteration,
                "buffer_sizes": buf_sizes,
                "buffer_total": total_buf_size,
                "strategy_buffer_size": len(strategy_buffer),
                "config": {
                    "architecture": "dueling",
                    "deployment": "average_strategy",
                    "feature_dim": FEATURE_DIM,
                    "trunk_dim": TRUNK_DIM,
                    "head_dim": HEAD_DIM,
                    "n_actions": N_ACTIONS,
                    "buffer_capacity_per_player": buffer_capacity,
                    "strategy_buffer_capacity": strategy_buffer_capacity,
                    "strategy_train_steps": strategy_train_steps,
                    "opponent_exploration_start": opponent_exploration_start,
                    "opponent_exploration_end": opponent_exploration_end,
                    "opponent_exploration_decay_iters": opponent_exploration_decay_iters,
                    "randomize_stacks": randomize_stacks,
                    "resume_buffers_saved": save_resume_buffers,
                    "strategy_variants": [
                        {"name": n, "steps": s, "trunk": t} for n, s, t in variants
                    ],
                    "export_advantage_npz": export_advantage_npz,
                },
            }
            torch.save(ckpt_data, save_path)

            latest_data = ckpt_data
            if save_resume_buffers:
                latest_data = dict(ckpt_data)
                latest_data["buffers"] = [b.state_dict() for b in buffers]
                latest_data["strategy_buffer"] = strategy_buffer.state_dict()
            torch.save(latest_data, resume_file)

            exported: list[str] = []
            for vname, _vsteps, _vtrunk in variants:
                snet = strategy_nets.get(vname)
                if snet is None:
                    continue
                vdir = ckpt_path if vname is None else Path(f"{ckpt_path}_{vname}")
                vdir.mkdir(parents=True, exist_ok=True)
                vpath = vdir / f"iter_{iteration:05d}.npz"
                _export_strategy_npz(snet, vpath)
                exported.append(str(vpath))
            if export_advantage_npz:
                adv_dir = Path(f"{ckpt_path}_adv")
                adv_dir.mkdir(parents=True, exist_ok=True)
                adv_path = adv_dir / f"iter_{iteration:05d}.npz"
                _export_advantage_npz(net, adv_path)
                exported.append(str(adv_path))

            print(f"  -> Saved checkpoint: {save_path} + {len(exported)} npz")

    print(
        f"\n[Deep CFR] Training complete. {n_iterations} iterations in {time.time() - start_time:.0f}s"
    )
    print(f"[Deep CFR] Final model: {ckpt_path / 'latest.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single Deep CFR training")
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--traversals", type=int, default=100_000)
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=5_000_000,
        help="Per-player buffer capacity (total = 6x this)",
    )
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--strategy-train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/deep_cfr")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel traversal workers (1=serial)",
    )
    parser.add_argument(
        "--strategy-buffer-capacity",
        type=int,
        default=5_000_000,
        help="Total reservoir capacity for average strategy samples",
    )
    parser.add_argument(
        "--opponent-exploration-start",
        type=float,
        default=0.10,
        help="Initial epsilon mixed into non-traverser action sampling",
    )
    parser.add_argument(
        "--opponent-exploration-end",
        type=float,
        default=0.02,
        help="Final epsilon mixed into non-traverser action sampling",
    )
    parser.add_argument(
        "--opponent-exploration-decay-iters",
        type=int,
        default=0,
        help="Linear decay horizon; 0 means all iterations",
    )
    parser.add_argument(
        "--no-stack-randomization",
        action="store_true",
        help="Disable match-stack sampling and train every hand at 10,000 chips",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from checkpoints that include replay buffers",
    )
    parser.add_argument(
        "--save-resume-buffers",
        action="store_true",
        help="Store replay buffers in latest.pt so --resume is exact",
    )
    parser.add_argument(
        "--strategy-variants",
        type=str,
        default="",
        help=(
            "Train multiple strategy read-outs on the same buffer for one-variable arm "
            "comparison. Format: 'name:steps:trunk,name:steps:trunk'. First is the control. "
            "Each exports to '<checkpoint-dir>_<name>'. Empty = single default strategy net."
        ),
    )
    parser.add_argument(
        "--export-advantage-npz",
        action="store_true",
        help="Also export the advantage net per checkpoint to '<checkpoint-dir>_adv' "
        "for strategy-net vs advantage-net deployment A/B.",
    )
    parser.add_argument(
        "--strategy-milestone-every",
        type=int,
        default=0,
        help="Every N iterations, log strategy CE at step milestones for the longest variant "
        "(0 = off). Answers whether the strategy net is still improving past 5K steps.",
    )
    args = parser.parse_args()

    strategy_variants = (
        _parse_strategy_variants(args.strategy_variants)
        if args.strategy_variants
        else None
    )

    train_deep_cfr(
        n_iterations=args.iterations,
        n_traversals=args.traversals,
        buffer_capacity=args.buffer_capacity,
        train_steps=args.train_steps,
        strategy_train_steps=args.strategy_train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        device_str=args.device,
        seed=args.seed,
        n_workers=args.workers,
        strategy_buffer_capacity=args.strategy_buffer_capacity,
        opponent_exploration_start=args.opponent_exploration_start,
        opponent_exploration_end=args.opponent_exploration_end,
        opponent_exploration_decay_iters=args.opponent_exploration_decay_iters,
        randomize_stacks=not args.no_stack_randomization,
        resume=args.resume,
        save_resume_buffers=args.save_resume_buffers,
        strategy_variants=strategy_variants,
        export_advantage_npz=args.export_advantage_npz,
        strategy_milestone_every=args.strategy_milestone_every,
    )


if __name__ == "__main__":
    main()
