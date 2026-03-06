VOCAB = {
    # Special tokens
    '[PAD]': 0,
    '[CLS]': 1,
    '[SEP]': 2,

    # Piece characters (12 pieces)
    'P': 3, 'N': 4, 'B': 5, 'R': 6, 'Q': 7, 'K': 8,      # white
    'p': 9, 'n': 10, 'b': 11, 'r': 12, 'q': 13, 'k': 14,  # black

    # Empty square
    '.': 15,

    # Side to move
    'w': 16, 'b_side': 17,  # 'b' for black side (distinct from bishop)

    # Castling characters
    'K_castle': 18, 'Q_castle': 19, 'k_castle': 20, 'q_castle': 21,
    '-': 22,  # no castling / no en passant

    # File letters (for en passant square)
    'a': 23, 'b_file': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28, 'g': 29, 'h': 30,

    # Rank digits (for en passant and move counters)
    '0': 31, '1': 32, '2': 33, '3': 34, '4': 35, '5': 36,
    '6': 37, '7': 38, '8': 39, '9': 40,
}

class FENTokenizer:
    def __init__(self):
        self.vocab = VOCAB
        self.seq_len = 80

    def tokenize(self, fen: str) -> list:
        parts = fen.split()
        board, side, castling, ep, halfmove, fullmove = parts

        tokens = [self.vocab['[CLS]']]

        # Board: 64 tokens
        for rank_str in board.split('/'):
            for ch in rank_str:
                if ch.isdigit():
                    tokens.extend([self.vocab['.']] * int(ch))
                else:
                    tokens.append(self.vocab[ch])  # piece char

        tokens.append(self.vocab['[SEP]'])

        # Side to move: 1 token
        tokens.append(self.vocab['w'] if side == 'w' else self.vocab['b_side'])

        # Castling: 4 tokens (K, Q, k, q presence)
        for ch, tok_name in [('K','K_castle'),('Q','Q_castle'),
                             ('k','k_castle'),('q','q_castle')]:
            tokens.append(self.vocab[tok_name] if ch in castling else self.vocab['-'])

        # En passant: 2 tokens
        if ep == '-':
            tokens.extend([self.vocab['-'], self.vocab['[PAD]']])
        else:
            file_char = ep[0]
            rank_char = ep[1]
            tokens.append(self.vocab.get(f'{file_char}_file', self.vocab[file_char]))
            tokens.append(self.vocab[rank_char])  # rank digit

        # Halfmove: 3 digits, zero-padded
        for ch in str(int(halfmove)).zfill(3):
            tokens.append(self.vocab[ch])

        # Fullmove: 4 digits, zero-padded
        for ch in str(int(fullmove)).zfill(4):
            tokens.append(self.vocab[ch])

        # Ensure we have exactly 80 tokens (truncate or pad if necessary, 
        # but logic above should yield exactly 80)
        if len(tokens) > 80:
            tokens = tokens[:80]
        elif len(tokens) < 80:
            tokens.extend([self.vocab['[PAD]']] * (80 - len(tokens)))

        return tokens
