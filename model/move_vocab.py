def build_move_vocab():
    # Returns (move_to_idx: dict, idx_to_move:list)
    moves = set()

    for src_rank in range(8):
        for src_file in range(8):
            src = f"{'abcdefgh'[src_file]}{src_rank+1}"

            # Rook/Queen directions (rank & file)
            for dr, df in [(1,0),(-1,0),(0,1),(0,-1)]:
                for dist in range(1, 8):
                    r, f = src_rank + dr*dist, src_file + df*dist
                    if 0 <= r < 8 and 0 <= f < 8:
                        dst = f"{'abcdefgh'[f]}{r + 1}"
                        moves.add(src + dst)

            # Bishop/Queen directions (diagonals)
            for dr, df in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                for dist in range(1, 8):
                    r, f = src_rank + dr*dist, src_file + df*dist
                    if 0 <= r < 8 and 0 <= f < 8:
                        dst = f"{'abcdefgh'[f]}{r + 1}"
                        moves.add(src + dst)

            # Knight moves
            for dr, df in [(2,1),(2,-1),(-2,1),(-2,-1),
                           (1,2),(1,-2),(-1,2),(-1,-2)]:
                r, f = src_rank + dr, src_file + df
                if 0 <= r < 8 and 0 <= f < 8:
                    dst = f"{'abcdefgh'[f]}{r + 1}"
                    moves.add(src + dst)
    # Add promotion suffixes
    promo_moves = set()
    for move in moves:
        src_rank = int(move[1]) - 1  # 0-indexed
        dst_rank = int(move[3]) - 1
        # White pawn promotes: src on rank 6 (7th rank), dst on rank 7 (8th rank)
        # Black pawn promotes: src on rank 1 (2nd rank), dst on rank 0 (1st rank)
        if (src_rank == 6 and dst_rank == 7) or (src_rank == 1 and dst_rank == 0):
            # Only add promotions for pawn-like moves (1 square forward, or 1 diagonal)
            file_diff = abs(ord(move[2]) - ord(move[0]))
            rank_diff = abs(dst_rank - src_rank)
            if rank_diff == 1 and file_diff <= 1:
                for suffix in ['q', 'r', 'b', 'n']:
                    promo_moves.add(move + suffix)

    moves = moves | promo_moves
    idx_to_move = sorted(moves)
    move_to_idx = {m: i for i, m in enumerate(idx_to_move)}
    return move_to_idx, idx_to_move
