"""Streamlit UI — ручной ввод game state + рекомендация.

Запуск:
    streamlit run poker_agent/ui.py
"""

from __future__ import annotations

import streamlit as st

from poker_agent.adviser import advise
from poker_agent.game_state import (
    Action,
    ActionKind,
    Card,
    GameState,
    Player,
    Position,
    parse_cards,
)

st.set_page_config(page_title="Poker Live Assistant", layout="wide")
st.title("Poker Live Assistant")
st.caption("Texas Hold'em — домашняя игра. NLHE 6-max, 100bb по умолчанию.")

# ---------- Sidebar: setup ----------

with st.sidebar:
    st.header("Сетап")
    n_players = st.slider("Игроков в раздаче", 2, 9, 6)
    hero_pos_str = st.selectbox(
        "Твоя позиция",
        [p.value for p in Position],
        index=3,  # BTN
    )
    effective_stack = st.number_input("Effective stack (bb)", 10.0, 500.0, 100.0, 5.0)

# ---------- Main: hand input ----------

col_hand, col_board = st.columns(2)

with col_hand:
    st.subheader("Твои карты")
    hero_str = st.text_input("Hole cards", "AsKs", help="Формат: AsKs или As Ks")

with col_board:
    st.subheader("Борд")
    board_str = st.text_input("Flop/Turn/River", "", help="Пусто = preflop. Пример: 7h2c9d")

# ---------- Action history ----------

st.subheader("Что было до тебя в раздаче")
col_pot, col_call = st.columns(2)
with col_pot:
    pot = st.number_input("Текущий pot (bb)", 0.0, 1000.0, 1.5, 0.5)
with col_call:
    to_call = st.number_input("Сколько надо доколлить (bb)", 0.0, 500.0, 0.0, 0.5)

action_summary = st.text_area(
    "Краткая история (опционально)",
    "",
    placeholder="Например: UTG raise 3bb, MP fold, CO 3bet 9bb",
)

# ---------- Get advice ----------

if st.button("ПОДУМАЙ", type="primary", use_container_width=True):
    try:
        hero_cards = parse_cards(hero_str)
        if len(hero_cards) != 2:
            st.error(f"Нужно ровно 2 карты, получено {len(hero_cards)}")
            st.stop()

        board_cards = parse_cards(board_str) if board_str.strip() else []
        if len(board_cards) not in (0, 3, 4, 5):
            st.error(f"Борд: 0/3/4/5 карт. Получено {len(board_cards)}")
            st.stop()

        # Соберём игроков. Hero на нужной позиции, остальные — заглушки.
        all_positions = list(Position)
        hero_pos = Position(hero_pos_str)
        # Простая логика: первые n_players позиций по 6-max порядку
        positions_in_play = all_positions[: n_players]
        if hero_pos not in positions_in_play:
            positions_in_play[-1] = hero_pos  # подменим последнюю

        players = [
            Player(
                position=p,
                stack_bb=effective_stack,
                in_hand=True,
                is_hero=(p == hero_pos),
            )
            for p in positions_in_play
        ]

        state = GameState(
            hero_cards=tuple(hero_cards),
            board=board_cards,
            players=players,
            pot_bb=pot,
            to_call_bb=to_call,
            effective_stack_bb=effective_stack,
            action_history=[
                # На M1 не парсим строку — солвер всё равно не подключён.
                # Для preflop adviser важен только факт "был ли raise до hero",
                # который определяем через to_call > 0. Подставляем позицию
                # отличную от hero, чтобы фильтр `a.player != hero_pos` сработал.
                Action(
                    player=next(
                        (p for p in positions_in_play if p != hero_pos),
                        Position.UTG,
                    ),
                    kind=ActionKind.RAISE,
                    amount_bb=to_call,
                )
            ]
            if to_call > 0
            else [],
        )

        rec = advise(state)

        # ---------- Отрисовка ----------
        st.divider()
        st.subheader("Рекомендация")

        m1, m2, m3 = st.columns(3)
        m1.metric("Стрит", rec["street"])
        m2.metric("Рука", rec["hand_code"])
        m3.metric("Pot odds", f"{rec['pot_odds']:.1%}")

        if rec.get("phase") == "RFI":
            color = "🟢" if rec["action"] == "raise" else "🔴"
            st.markdown(f"### {color} **{rec['action'].upper()}**")
            if rec["action"] == "raise":
                st.markdown(f"Размер открытия: **{rec['size_bb']} bb**")
            st.caption(rec["reasoning"])

        elif rec.get("phase") == "facing_raise":
            st.warning(rec["action"])
            st.caption(rec["reasoning"])

        elif rec.get("phase") == "postflop":
            eq_col, odds_col = st.columns(2)
            eq_col.metric("Equity vs random", f"{rec['equity_vs_random']:.1%}")
            odds_col.metric(
                "EV call",
                "+" if rec["equity_vs_random"] > rec["pot_odds"] else "-",
                delta=f"{(rec['equity_vs_random'] - rec['pot_odds']):.1%}",
            )
            st.caption(rec["reasoning"])

            solver_info = rec.get("solver", {})
            if solver_info.get("available"):
                st.success("TexasSolver:")
                st.json(solver_info["best_action"])
            else:
                st.info("ℹ️ " + solver_info.get("msg", "solver не подключён"))

        with st.expander("Raw output"):
            st.json(rec)

        if action_summary.strip():
            st.caption(f"📝 История: {action_summary}")

    except Exception as e:  # noqa: BLE001
        st.error(f"Ошибка: {e}")
        raise

st.divider()
with st.expander("Статус модулей"):
    from poker_agent import camera as cam_mod
    from poker_agent import detector as det_mod
    from poker_agent import solver as solv_mod

    st.write("**Camera (M4):** заглушка")
    if cam_mod.cv2 is not None:
        try:
            st.write(f"  доступные камеры: {cam_mod.list_cameras(3)}")
        except Exception as e:  # noqa: BLE001
            st.write(f"  ошибка: {e}")
    else:
        st.write("  opencv не установлен")

    st.write(f"**Detector (M3):** веса {'есть' if det_mod.DEFAULT_MODEL_PATH.exists() else 'НЕТ'}")
    st.write(f"**Solver (M2):** {'установлен' if solv_mod.is_available() else 'НЕТ'}")
