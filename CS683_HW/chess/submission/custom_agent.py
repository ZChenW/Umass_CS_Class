import random
import time
import chess as ch


class CustomAgent:
    """
    Example random agent. Replace takeAction() with your own logic.

    Your agent will be instantiated once and reused across games.
    """

    INF = 10**12
    MATE = 10**9

    def __init__(self):
        self.deadline = 0.0
        self.nodes = 0

    def takeAction(self, state):
        """
        Select an action for the given game state.

        Parameters:
            state : State
                Current game state. Key methods/attributes:
                  state.getActions()  -> list of legal actions
                  state.player        -> 0 (MAX player) or 1 (MIN player)
                  state.takeAction(a) -> new State after applying action a
                  state.isTerminal()  -> True if game is over
                  state.getValue()    -> terminal value (+1 / -1 / 0)
                  state.Evaluate()    -> heuristic score estimate

        Returns:
            action : one element from state.getActions()
        """
        actions = state.getActions()
        if not actions:
            return None

        board = state.board
        self.deadline = time.perf_counter() + 0.70
        self.nodes = 0

        mate = self._mate_in_one(board, actions)
        if mate is not None:
            return mate

        best_move = self._greedy_move(board, actions)

        if len(actions) <= 10:
            max_depth = 4
        else:
            max_depth = 3

        for depth in range(1, max_depth + 1):
            result = self._root_search(board, depth, best_move)
            if result is None:
                break

            move, score = result
            if move is not None:
                best_move = move

            if abs(score) > self.MATE - 10000:
                break

        return best_move

    def _time_up(self):
        self.nodes += 1
        if self.nodes % 12 == 0:
            return time.perf_counter() >= self.deadline
        return False

    def _root_search(self, board, depth, previous_best):
        best_move = previous_best
        best_score = -self.INF
        alpha = -self.INF
        beta = self.INF

        moves = self._order_moves(board, list(board.legal_moves), previous_best)

        for move in moves:
            if self._time_up():
                return None

            board.push(move)
            score = self._alphabeta(board, depth - 1, -beta, -alpha, 1)
            board.pop()

            if score is None:
                return None

            score = -score

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        return best_move, best_score

    def _alphabeta(self, board, depth, alpha, beta, ply):
        if self._time_up():
            return None

        if board.is_game_over(claim_draw=True):
            return self._terminal_score(board, ply)

        if depth <= 0:
            if board.is_check():
                depth = 1
            else:
                return self._quiescence(board, alpha, beta, ply, 2)

        best = -self.INF
        moves = self._order_moves(board, list(board.legal_moves), None)

        for move in moves:
            board.push(move)
            score = self._alphabeta(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score is None:
                return None

            score = -score

            if score > best:
                best = score

            if best > alpha:
                alpha = best

            if alpha >= beta:
                break

        return best

    def _quiescence(self, board, alpha, beta, ply, qdepth):
        if self._time_up():
            return None

        if board.is_game_over(claim_draw=True):
            return self._terminal_score(board, ply)

        stand_pat = self._side_to_move_eval(board)

        if stand_pat >= beta:
            return beta

        if stand_pat > alpha:
            alpha = stand_pat

        if qdepth <= 0:
            return alpha

        moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion:
                moves.append(move)

        moves = self._order_moves(board, moves, None)

        for move in moves:
            board.push(move)
            score = self._quiescence(board, -beta, -alpha, ply + 1, qdepth - 1)
            board.pop()

            if score is None:
                return None

            score = -score

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

    def _terminal_score(self, board, ply):
        if board.is_checkmate():
            return -self.MATE + ply
        return 0

    def _mate_in_one(self, board, actions):
        for move in actions:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()

            if is_mate:
                return move

        return None

    def _greedy_move(self, board, actions):
        best_moves = []
        best_score = -self.INF

        for move in self._order_moves(board, actions, None):
            board.push(move)
            score = -self._side_to_move_eval(board)
            board.pop()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def _order_moves(self, board, moves, preferred):
        def move_score(move):
            score = 0

            if preferred is not None and move == preferred:
                score += 100000

            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)

                if victim is None and board.is_en_passant(move):
                    victim_value = self._piece_value(ch.PAWN)
                else:
                    victim_value = self._piece_value(victim.piece_type) if victim else 0

                attacker_value = (
                    self._piece_value(attacker.piece_type) if attacker else 0
                )
                score += 10000 + 12 * victim_value - attacker_value

            if move.promotion:
                score += 9000 + self._piece_value(move.promotion)

            if board.gives_check(move):
                score += 1200

            if board.is_castling(move):
                score += 350

            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [ch.KNIGHT, ch.BISHOP]:
                if move.from_square in [
                    ch.B1,
                    ch.G1,
                    ch.B8,
                    ch.G8,
                    ch.C1,
                    ch.F1,
                    ch.C8,
                    ch.F8,
                ]:
                    score += 100

            file = ch.square_file(move.to_square)
            rank = ch.square_rank(move.to_square)
            score += 20 - 4 * (abs(file - 3.5) + abs(rank - 3.5))

            return score

        return sorted(moves, key=move_score, reverse=True)

    def _side_to_move_eval(self, board):
        score = self._evaluate_white(board)

        if board.turn == ch.WHITE:
            return score
        return -score

    def _evaluate_white(self, board):
        if board.is_checkmate():
            if board.turn == ch.WHITE:
                return -self.MATE
            return self.MATE

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0

        for square, piece in board.piece_map().items():
            sign = 1 if piece.color == ch.WHITE else -1
            score += sign * self._piece_value(piece.piece_type)
            score += sign * self._square_bonus(piece, square)

        score += self._center_score(board, ch.WHITE)
        score -= self._center_score(board, ch.BLACK)

        score += self._king_safety(board, ch.WHITE)
        score -= self._king_safety(board, ch.BLACK)

        score += self._simple_threats(board, ch.WHITE)
        score -= self._simple_threats(board, ch.BLACK)

        score += self._development(board, ch.WHITE)
        score -= self._development(board, ch.BLACK)

        if board.turn == ch.WHITE:
            score += 2 * len(list(board.legal_moves))
        else:
            score -= 2 * len(list(board.legal_moves))

        if board.is_check():
            if board.turn == ch.WHITE:
                score -= 60
            else:
                score += 60

        return score

    def _piece_value(self, piece_type):
        values = {
            ch.PAWN: 100,
            ch.KNIGHT: 320,
            ch.BISHOP: 330,
            ch.ROOK: 500,
            ch.QUEEN: 900,
            ch.KING: 0,
        }

        return values.get(piece_type, 0)

    def _square_bonus(self, piece, square):
        if piece.color == ch.BLACK:
            square = ch.square_mirror(square)

        file = ch.square_file(square)
        rank = ch.square_rank(square)
        center = 14 - 4 * (abs(file - 3.5) + abs(rank - 3.5))

        if piece.piece_type == ch.PAWN:
            return 6 * rank + center

        if piece.piece_type == ch.KNIGHT:
            edge = int(file in [0, 7]) + int(rank in [0, 7])
            return 4 * center - 14 * edge

        if piece.piece_type == ch.BISHOP:
            return 3 * center + rank

        if piece.piece_type == ch.ROOK:
            return 2 * rank

        if piece.piece_type == ch.QUEEN:
            return 2 * center

        if piece.piece_type == ch.KING:
            return -2 * center

        return 0

    def _center_score(self, board, color):
        total = 0

        for square in [ch.D4, ch.E4, ch.D5, ch.E5]:
            total += 8 * len(board.attackers(color, square))

            piece = board.piece_at(square)
            if piece and piece.color == color:
                total += 15

        return total

    def _king_safety(self, board, color):
        king = board.king(color)
        if king is None:
            return -500

        total = 0
        king_file = ch.square_file(king)
        king_rank = ch.square_rank(king)

        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                if df == 0 and dr == 0:
                    continue

                file = king_file + df
                rank = king_rank + dr

                if 0 <= file <= 7 and 0 <= rank <= 7:
                    square = ch.square(file, rank)
                    piece = board.piece_at(square)

                    if piece and piece.color == color and piece.piece_type == ch.PAWN:
                        total += 14

        total -= 22 * len(board.attackers(not color, king))
        return total

    def _simple_threats(self, board, color):
        total = 0
        enemy = not color

        for square, piece in board.piece_map().items():
            if piece.color == enemy and piece.piece_type != ch.KING:
                if board.is_attacked_by(color, square):
                    total += self._piece_value(piece.piece_type) // 8

        return total

    def _development(self, board, color):
        total = 0

        if color == ch.WHITE:
            home_knights = [ch.B1, ch.G1]
            home_bishops = [ch.C1, ch.F1]
        else:
            home_knights = [ch.B8, ch.G8]
            home_bishops = [ch.C8, ch.F8]

        for square in board.pieces(ch.KNIGHT, color):
            if square not in home_knights:
                total += 12

        for square in board.pieces(ch.BISHOP, color):
            if square not in home_bishops:
                total += 10

        return total
