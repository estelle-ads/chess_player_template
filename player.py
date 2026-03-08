import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Qwen baseline chess player.

    REQUIRED:
        Subclasses chess_tournament.players.Player
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "Qwen2.5-1.5B-Instruct", ### NEW
        model_id: str = "HuggingFaceTB/Qwen/Qwen2.5-1.5B-Instruct", #NEW: Qwen2.5-1.5B
        temperature: float = 0.2,  ### NEW: lower temperature for stable moves
        max_new_tokens: int = 8,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded components
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            print(f"[{self.name}] Loading {self.model_id} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt (adjusted part, see commented parts!!!)
    # -------------------------
    ### NEW def _build_prompt
    def _build_prompt(self, fen: str) -> str:

        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        return f"""You are a chess engine. Choose the BEST move from the list of legal moves.
        Output ONE move, exactly in UCI format (e.g., e2e4, g1f3, e7e8q).
        Do NOT output explanations, punctuation, or extra text.

Legal moves: {legal_moves}
Position: {fen}
Best move:
"""
    ### END of the new part^

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        try:
            self._load_model()
        except Exception:
            return self._random_legal(fen)

        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        prompt = self._build_prompt(fen)

        ### NEW: majority vote
        move_counts = {}

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate multiple samples
            for _ in range(5):  ### NEW: 5 samples for majority vote
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt):]

                move = self._extract_move(decoded)

                # Count only legal moves
                if move and move in legal_moves:
                    move_counts[move] = move_counts.get(move, 0) + 1

            # Return the most frequent legal move
            if move_counts:
                return max(move_counts, key=move_counts.get)

        except Exception:
            pass

        ### fallback to random legal move
        return random.choice(legal_moves)
    ### END NEW majority vote
