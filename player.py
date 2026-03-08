import os
import sys
import json
import torch
import chess

# Ensure the repo root is on sys.path so "model" package is importable
# when this file is loaded via importlib from an external working directory.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from model.tokenizer import FENTokenizer
from model.move_vocab import build_move_vocab
from model.architecture import ChessTransformer

# Import Player base class from teacher's chess_tournament if available,
# otherwise define a compatible fallback so the file works standalone.
try:
    from chess_tournament import Player
except ImportError:
    from abc import ABC, abstractmethod
    from typing import Optional

    class Player(ABC):
        def __init__(self, name: str):
            self.name = name

        @abstractmethod
        def get_move(self, fen: str) -> Optional[str]:
            pass


# ---- Opening Book ----
# Strong mainline openings (UCI move sequences).
# We follow the book as long as the game matches a known line.
OPENING_BOOK = {
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": "e2e4",
    # Sicilian Defense
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": "c7c5",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "g1f3",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": "d7d6",
    "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3": "d2d4",
    # Italian Game
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "g1f3",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": "b8c6",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": "f1c4",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "f8c5",
    # Queen's Gambit (if opponent plays d5)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": "e7e5",
    # French Defense response
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "d2d4",
    # Caro-Kann response
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "d2d4",
    # As black against d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": "g8f6",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": "c2c4",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": "e7e6",
    # As black against Nf3
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1": "d7d5",
    # As black against c4 (English)
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1": "e7e5",
}

# ---- Piece values for heuristics ----
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class TransformerPlayer(Player):
    def __init__(self, name: str = "Chess-Bot-20M"):
        super().__init__(name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenizer and move vocab
        self.tokenizer = FENTokenizer()
        self.move_to_idx, self.idx_to_move = build_move_vocab()

        # Load model from checkpoint
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        config_path = os.path.join(checkpoint_dir, "best_config.json")
        weights_path = os.path.join(checkpoint_dir, "best.pt")

        with open(config_path) as f:
            cfg = json.load(f)

        self.model = ChessTransformer(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            seq_len=cfg["seq_len"],
            n_moves=cfg["n_moves"],
            dropout=0.0,
        )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        # Strip torch.compile prefix if present
        state_dict = {k.replace("_orig_mod.", ""): v.float() for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Syzygy tablebase (optional, used if path exists)
        self.tablebase = None
        tb_path = os.path.join(os.path.dirname(__file__), "syzygy")
        if os.path.isdir(tb_path):
            try:
                self.tablebase = chess.syzygy.open_tablebase(tb_path)
            except Exception:
                pass

    @torch.no_grad()
    def _get_model_scores(self, fen: str):
        """Return raw logits for a position."""
        tokens = self.tokenizer.tokenize(fen)
        token_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model(token_ids)[0]  # (1968,)

    def _get_legal_move_scores(self, board: chess.Board, logits: torch.Tensor):
        """Get list of (move_uci, score) for all legal moves, sorted by score descending."""
        legal_moves = {m.uci() for m in board.legal_moves}
        scored = []
        for uci in legal_moves:
            idx = self.move_to_idx.get(uci)
            if idx is not None:
                scored.append((uci, logits[idx].item()))
            else:
                scored.append((uci, -999.0))
        scored.sort(key=lambda x: -x[1])
        return scored

    def _is_blunder(self, board: chess.Board, move_uci: str):
        """Check if a move hangs a piece or allows mate in 1."""
        move = chess.Move.from_uci(move_uci)
        board.push(move)
        try:
            # Check if we just allowed mate in 1
            if board.is_checkmate():
                return True

            for resp in board.legal_moves:
                if board.is_capture(resp):
                    # Opponent can capture something after our move
                    captured = board.piece_at(resp.to_square)
                    if captured:
                        # Check if the captured piece is undefended or high value
                        moving = board.piece_at(resp.from_square)
                        if moving and captured:
                            cap_val = PIECE_VALUES.get(captured.piece_type, 0)
                            atk_val = PIECE_VALUES.get(moving.piece_type, 0)
                            # Hanging piece: we lose more than a pawn net
                            if cap_val >= 3 and cap_val > atk_val:
                                return True

                # Check if opponent has mate in 1
                board.push(resp)
                if board.is_checkmate():
                    board.pop()
                    return True
                board.pop()

            return False
        finally:
            board.pop()

    def _score_with_heuristics(self, board: chess.Board, move_uci: str, model_score: float):
        """Adjust model score with chess heuristics."""
        move = chess.Move.from_uci(move_uci)
        score = model_score

        # Bonus for giving check
        board.push(move)
        if board.is_check():
            score += 0.3
        if board.is_checkmate():
            score += 100.0  # Always play checkmate
        board.pop()

        # Bonus for captures (proportional to captured piece value)
        if board.is_capture(move):
            captured_sq = move.to_square
            captured = board.piece_at(captured_sq)
            if captured:
                score += PIECE_VALUES.get(captured.piece_type, 0) * 0.05

            # Extra bonus for capturing with lower-value piece (good trade)
            attacker = board.piece_at(move.from_square)
            if captured and attacker:
                trade_gain = PIECE_VALUES.get(captured.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0)
                if trade_gain > 0:
                    score += trade_gain * 0.1

        # Bonus for promotion
        if move.promotion:
            score += 2.0

        return score

    def _search_1ply(self, board: chess.Board, candidates):
        """
        1-ply lookahead: for each candidate move, evaluate the resulting
        position from the opponent's perspective. Pick the move that gives
        the opponent the lowest best response.

        candidates: list of (move_uci, score) — top moves to search
        """
        best_move = candidates[0][0]
        best_value = float('-inf')

        for move_uci, base_score in candidates:
            move = chess.Move.from_uci(move_uci)
            board.push(move)

            if board.is_checkmate():
                board.pop()
                return move_uci  # Instant win

            if board.is_stalemate() or board.is_insufficient_material():
                # Draw — neutral
                opp_best_score = 0.0
            elif not list(board.legal_moves):
                opp_best_score = 0.0
            else:
                opp_logits = self._get_model_scores(board.fen())
                opp_scored = self._get_legal_move_scores(board, opp_logits)
                opp_best_score = opp_scored[0][1] if opp_scored else 0.0

            board.pop()

            # We want to minimize opponent's best response
            # Combined: our model score minus opponent's best
            value = base_score - opp_best_score * 0.5

            if value > best_value:
                best_value = value
                best_move = move_uci

        return best_move

    def _tablebase_move(self, board: chess.Board):
        """Try to get a perfect move from Syzygy tablebases (<=5 pieces)."""
        if self.tablebase is None:
            return None
        if chess.popcount(board.occupied) > 5:
            return None
        try:
            best_move = None
            best_wdl = -3
            for move in board.legal_moves:
                board.push(move)
                # WDL from opponent's perspective after our move, so negate
                wdl = -self.tablebase.probe_wdl(board)
                if wdl > best_wdl:
                    best_wdl = wdl
                    best_move = move
                board.pop()
            if best_move:
                return best_move.uci()
        except Exception:
            pass
        return None

    def _check_forced_mate(self, board: chess.Board):
        """If any legal move is checkmate, play it immediately."""
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()
        return None

    @torch.no_grad()
    def get_move(self, fen: str):
        board = chess.Board(fen)

        # 0. Forced mate-in-1: always play checkmate if available
        mate_move = self._check_forced_mate(board)
        if mate_move:
            return mate_move

        # 1. Opening book
        if fen in OPENING_BOOK:
            book_move = OPENING_BOOK[fen]
            # Verify it's legal (safety check)
            try:
                if chess.Move.from_uci(book_move) in board.legal_moves:
                    return book_move
            except Exception:
                pass

        # 2. Endgame tablebase (perfect play with <=5 pieces)
        tb_move = self._tablebase_move(board)
        if tb_move:
            return tb_move

        # 3. Get model predictions
        logits = self._get_model_scores(fen)
        scored_moves = self._get_legal_move_scores(board, logits)

        if not scored_moves:
            return None

        # 4. Apply heuristic adjustments (check/capture/promotion bonuses)
        adjusted = []
        for move_uci, raw_score in scored_moves:
            adj_score = self._score_with_heuristics(board, move_uci, raw_score)
            adjusted.append((move_uci, adj_score))
        adjusted.sort(key=lambda x: -x[1])

        # 5. Blunder detection on top pick
        top_move = adjusted[0][0]
        if len(adjusted) > 1 and self._is_blunder(board, top_move):
            # Try next candidates
            for move_uci, _ in adjusted[1:5]:
                if not self._is_blunder(board, move_uci):
                    top_move = move_uci
                    break

        # 6. 1-ply search on top 5 candidates (if enough legal moves)
        if len(adjusted) >= 3:
            top_candidates = adjusted[:5]
            top_move = self._search_1ply(board, top_candidates)

        return top_move
