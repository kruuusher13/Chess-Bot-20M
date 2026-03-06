# Chess-Bot-20M

A 39M-parameter encoder-only transformer that plays chess by predicting the best move from a FEN position. Trained on 20M positions from Lichess Stockfish evaluations.

## Architecture

```
Input: FEN string
  |
  v
FEN Tokenizer (80 fixed-length tokens)
  |  - 64 board squares + [CLS]/[SEP] + side/castling/en passant/clocks
  v
ChessTransformer (encoder-only, ~39M params)
  |  - Token + positional embeddings
  |  - 12 transformer blocks (pre-norm, 512-dim, 8 heads, 2048 FFN, GELU)
  |  - [CLS] token pooling
  v
Classification head -> 1968 possible UCI moves
  |
  v
Inference pipeline:
  1. Forced mate-in-1 check
  2. Opening book lookup
  3. Syzygy tablebase (<=5 pieces, optional)
  4. Model prediction + legal move masking
  5. Heuristic adjustments (check/capture/promotion bonuses)
  6. Blunder detection (hanging pieces, allowing mate-in-1)
  7. 1-ply lookahead on top 5 candidates
```

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden dim | 512 |
| Attention heads | 8 |
| FFN dim | 2048 |
| Vocab size | 51 |
| Output classes | 1968 (UCI moves) |
| Parameters | ~39M |

## Training

- **Data:** 10M Lichess Stockfish evaluations + 10M color-flipped augmentations (20M total)
- **Batch size:** 2048, **Steps:** 50,000 (~5.2 epochs)
- **Optimizer:** AdamW (lr=3e-4, warmup 2000 steps, cosine decay)
- **Hardware:** RunPod RTX 6000 Ada 48GB (~9.3 hours)
- **Final accuracy:** ~50% top-1, ~78% top-3

## Setup

```bash
git clone https://github.com/kruuusher13/Chess-Bot-20M.git
cd Chess-Bot-20M
pip install -r requirements.txt
```

## Play a Game

```bash
python -c "
from player import TransformerPlayer
import chess

player = TransformerPlayer('ChessBot')
board = chess.Board()

while not board.is_game_over():
    print(board)
    print()
    if board.turn == chess.WHITE:
        move = input('Your move (UCI, e.g. e2e4): ')
        try:
            board.push(chess.Move.from_uci(move))
        except:
            print('Invalid move, try again.')
            continue
    else:
        move = player.get_move(board.fen())
        print(f'Bot plays: {move}')
        board.push(chess.Move.from_uci(move))

print(board)
print('Result:', board.result())
"
```

## Project Structure

```
player.py              # Inference player (loads checkpoint, plays moves)
model/
  architecture.py      # ChessTransformer (encoder-only, ~39M params)
  tokenizer.py         # FEN -> 80 token IDs
  move_vocab.py        # 1968 UCI move vocabulary
checkpoints/
  best.pt              # Trained weights
  best_config.json     # Model config
```
