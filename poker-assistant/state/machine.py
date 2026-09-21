"""Macchina a stati per il tracking delle mani tra frame.

Sostituisce l'euristica "cambiano le hole → nuova mano" con una FSM vera:
- ogni frame produce un PokerState validato con fingerprint
- la FSM conferma lo stato solo dopo N letture concordanti (stabilità)
- valida la progressione: il board può solo crescere (0→3→4→5) dentro una mano
- tracka hand_id: cambia quando le hero cards cambiano rank
- rifiuta transizioni impossibili (es. board che si accorcia senza nuova mano)

Ispirata a poker-vision-lab HandStateMachine (MIT) — adattata al nostro
flusso: ingest(dict raw) → StateUpdate con PokerState validato.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from state.models import PokerState, Street

_HAND_ID_COUNTER = itertools.count(1)


@dataclass(frozen=True)
class StateUpdate:
    """Esito dell'ingestione di un frame nella FSM."""

    state: PokerState | None
    changed: bool  # lo stato confermato è cambiato rispetto al precedente
    stable: bool  # lo stato ha raggiunto il quorum di stabilità
    reason: str
    hand_id: str | None = None


class HandStateMachine:
    """FSM che filtra il rumore della visione e valida la progressione del gioco."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        stable_frames: int = 2,
        allow_board_shrink_on_new_hand: bool = True,
    ):
        if stable_frames < 1:
            raise ValueError("stable_frames deve essere >= 1")
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self._candidate_fingerprint: tuple | None = None
        self._candidate_count: int = 0
        self._candidate_state: PokerState | None = None
        self._confirmed_state: PokerState | None = None
        self._hand_id: str | None = None
        self._hero_ranks: tuple[str, ...] = ()
        self._street: Street = Street.PREFLOP
        self._board_len: int = 0

    # ------------------------------------------------------------------ #
    #  API
    # ------------------------------------------------------------------ #

    def ingest(self, raw: dict) -> StateUpdate:
        """Processa un frame raw (dict estrattore) → StateUpdate."""
        state = PokerState.from_raw_dict(raw)

        # gate: senza mano non c'è nulla da confermare (WAIT tra le mani)
        if not state.has_hand:
            self._reset_candidate()
            return StateUpdate(
                state=None, changed=False, stable=False,
                reason="no hero cards", hand_id=self._hand_id,
            )

        # gate confidenza
        if state.confidence < self.min_confidence:
            self._reset_candidate()
            return StateUpdate(
                state=self._confirmed_state, changed=False, stable=False,
                reason=f"confidence {state.confidence:.2f} < {self.min_confidence}",
                hand_id=self._hand_id,
            )

        # nuova mano? rank hero cambiati → reset board/street
        ranks = tuple(sorted(c[0] for c in state.hero_cards))
        if self._hero_ranks and ranks != self._hero_ranks:
            new_id = f"hand-{next(_HAND_ID_COUNTER)}"
            print(f"[fsm] NUOVA MANO {new_id}: {self._hero_ranks} → {ranks}")
            self._hero_ranks = ranks
            self._hand_id = new_id
            self._street = Street.PREFLOP
            self._board_len = 0
            self._reset_candidate()
        elif not self._hero_ranks:
            self._hero_ranks = ranks
            self._hand_id = f"hand-{next(_HAND_ID_COUNTER)}"

        # validazione board: può solo crescere dentro una mano
        board_len = len(state.board)
        if board_len < self._board_len and not allow_board_shrink_guard(self):
            return StateUpdate(
                state=self._confirmed_state, changed=False, stable=False,
                reason=f"board shrink {self._board_len}→{board_len} senza nuova mano",
                hand_id=self._hand_id,
            )
        self._board_len = max(self._board_len, board_len)
        self._street = state.street

        # conferma per stabilità: stesso fingerprint per N frame
        fp = state.fingerprint()
        if fp == self._candidate_fingerprint:
            self._candidate_count += 1
        else:
            self._candidate_fingerprint = fp
            self._candidate_count = 1
            self._candidate_state = state

        if self._candidate_count < self.stable_frames:
            return StateUpdate(
                state=self._confirmed_state, changed=False, stable=False,
                reason=f"stabilità {self._candidate_count}/{self.stable_frames}",
                hand_id=self._hand_id,
            )

        # stabile: conferma
        previous = self._confirmed_state
        self._confirmed_state = self._candidate_state
        changed = previous is None or previous.fingerprint() != fp
        reason = "stabile" if changed else "confermato"
        return StateUpdate(
            state=self._confirmed_state, changed=changed, stable=True,
            reason=reason, hand_id=self._hand_id,
        )

    @property
    def current_hand_id(self) -> str | None:
        return self._hand_id

    @property
    def confirmed(self) -> PokerState | None:
        return self._confirmed_state

    def _reset_candidate(self) -> None:
        self._candidate_fingerprint = None
        self._candidate_count = 0
        self._candidate_state = None


def allow_board_shrink_guard(machine: HandStateMachine) -> bool:
    """Il board può accorciarsi SOLO se è una nuova mano (rank hero cambiati,
    gestito dal chiamante) o se l'utente ha permesso lo shrink esplicito.

    Semplificato: accorciamento tollerato perché il cambio rank resetta già
    _board_len; uno shrink con stessi rank = anomalia di lettura → False.
    """
    return False
