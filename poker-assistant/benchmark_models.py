"""Benchmark modelli vision su OpenRouter per la lettura del tavolo poker.

Replica i parametri di produzione di llm_vision_extractor:
  - JPEG quality 92, max_dim 1280
  - temperature 0.1
  - SYSTEM_PROMPT identico

Per ogni modello × immagine esegue N run sequenziali, misura:
  - latenza wall-clock
  - validità JSON
  - stato estratto (consenso tra modelli = ground truth approssimato)

Uso:  python3 benchmark_models.py
"""
import importlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "vision"))
_lve = importlib.import_module("llm_vision_extractor")
_encode_frame_to_base64 = _lve._encode_frame_to_base64
SYSTEM_PROMPT = _lve.SYSTEM_PROMPT

API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY mancante (.env)")

MODELS = [
    "google/gemini-3.1-flash-lite",   # baseline attuale
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.6-flash",
    "google/gemini-3.8-flash",
    "openai/gpt-5.6-luna",
]
IMAGES = ["calib_frame.png", "auto_calib_debug.png"]
RUNS = 3
PRICING = {  # $ per 1M token (in, out)
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "google/gemini-3.6-flash": (0.75, 3.75),
    "google/gemini-3.8-flash": (0.75, 3.75),
    "openai/gpt-5.6-luna": (0.20, 1.20),
}


def parse_state(text):
    """Estrae il JSON dal testo (tollera fence markdown)."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_model(model, b64):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the full poker table state from this screenshot."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    t0 = time.time()
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60.0)
    dt = time.time() - t0
    if not resp.ok:
        return {"ok": False, "latency": dt, "error": resp.text[:150]}
    data = resp.json()
    choice = data.get("choices", [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage", {})
    state = parse_state(content)
    return {
        "ok": state is not None,
        "latency": dt,
        "state": state,
        "raw": content[:200],
        "usage": {
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage.get("completion_tokens", 0),
        },
    }


def normalize_field(state, key):
    if state is None:
        return None
    v = state.get(key)
    if key in ("hole",):
        return tuple(sorted(v)) if isinstance(v, list) and len(v) == 2 else None
    if key in ("board",):
        return tuple(v) if isinstance(v, list) else None
    return v


FIELDS = ["hole", "board", "pot", "to_call", "stack", "stage", "button_seat"]


def main():
    results = {}  # (model, img) -> list of run dicts
    for img_name in IMAGES:
        frame = cv2.imread(str(ROOT / img_name))
        if frame is None:
            print(f"!! {img_name} non trovata, salto")
            continue
        b64 = _encode_frame_to_base64(frame)
        print(f"\n### {img_name} ({len(b64)//1024}KB payload)")
        for model in MODELS:
            runs = []
            for i in range(RUNS):
                r = call_model(model, b64)
                runs.append(r)
                tag = "OK " if r["ok"] else "ERR"
                extra = "" if r["ok"] else f" :: {r.get('error', r.get('raw', ''))[:80]}"
                print(f"  {tag} {model:32s} run{i+1} {r['latency']:5.2f}s{extra}")
                time.sleep(0.4)
            results[(model, img_name)] = runs

    # ---- consenso per campo (per immagine) ----
    print("\n\n================ RISULTATI ================")
    summary = {}
    for model in MODELS:
        lat, ok_cnt, tot, agree, stable = [], 0, 0, 0.0, 0
        tok_in, tok_out = 0, 0
        for img_name in IMAGES:
            runs = results.get((model, img_name), [])
            all_runs_img = [rr for mm in MODELS for rr in results.get((mm, img_name), []) if rr["ok"]]
            consensus = {}
            for f in FIELDS:
                vals = [normalize_field(rr["state"], f) for rr in all_runs_img]
                vals = [v for v in vals if v is not None]
                if vals:
                    consensus[f] = statistics.mode(vals)
            for rr in runs:
                tot += 1
                if rr["ok"]:
                    ok_cnt += 1
                    lat.append(rr["latency"])
                    tok_in += rr["usage"]["prompt"]
                    tok_out += rr["usage"]["completion"]
                    matches = sum(1 for f in FIELDS
                                  if normalize_field(rr["state"], f) is not None
                                  and normalize_field(rr["state"], f) == consensus.get(f))
                    denom = sum(1 for f in FIELDS if consensus.get(f) is not None)
                    agree += matches / max(denom, 1)
            # stabilità: stessa risposta su tutte le run dell'immagine
            ok_states = [json.dumps({f: normalize_field(rr["state"], f) for f in FIELDS}, default=str)
                         for rr in runs if rr["ok"]]
            if ok_states and len(set(ok_states)) == 1:
                stable += 1
        n_ok = max(ok_cnt, 1)
        pin, pout = PRICING[model]
        cost = (tok_in * pin + tok_out * pout) / 1e6
        try:
            lat_avg = statistics.mean(lat) if lat else float("nan")
            lat_min = min(lat) if lat else float("nan")
        except (ValueError, statistics.StatisticsError):
            lat_avg = lat_min = float("nan")
        summary[model] = {
            "latency_avg": lat_avg,
            "latency_min": lat_min,
            "valid_pct": 100 * ok_cnt / max(tot, 1),
            "consensus_pct": 100 * agree / n_ok,
            "stability": f"{stable}/{len(IMAGES)}",
            "cost_total": cost,
            "tok_per_call": (tok_in + tok_out) / n_ok,
        }

    hdr = f"{'modello':32s} {'lat avg':>8s} {'lat min':>8s} {'valid%':>7s} {'consenso%':>9s} {'stabile':>8s} {'$/1000call':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for model, s in sorted(summary.items(), key=lambda kv: kv[1]["latency_avg"]):
        cost_1k = s["cost_total"] / max(len(IMAGES) * RUNS, 1) * 1000
        print(f"{model:32s} {s['latency_avg']:7.2f}s {s['latency_min']:7.2f}s {s['valid_pct']:6.0f} {s['consensus_pct']:8.0f} {s['stability']:>8s} {cost_1k:10.2f}")

    # dettaglio estrazioni per immagine
    print("\n--- Estrazioni (run migliore per immagine) ---")
    for img_name in IMAGES:
        print(f"\n{img_name}:")
        for model in MODELS:
            runs = results.get((model, img_name), [])
            ok_runs = [rr for rr in runs if rr["ok"]]
            if ok_runs:
                st = ok_runs[0]["state"]
                comp = {k: st.get(k) for k in ("hole", "board", "pot", "to_call", "stack", "stage", "button_seat")}
                print(f"  {model:32s} {comp}")
            else:
                print(f"  {model:32s} NESSUNA ESTRAZIONE VALIDA")


if __name__ == "__main__":
    main()
