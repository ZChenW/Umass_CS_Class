import sys
from pathlib import Path

import chess as ch

sys.path.insert(0, str(Path(__file__).parent.parent))
from state import State
from algorithms import minimaxAlphaBetaRecursiveWithEvaluationFunction, MCTS_best_action


class ChessGame(State):
    """
    Chess game state wrapping the python-chess library.

    The MAX player controls White; the MIN player controls Black.
    The board follows standard chess rules including en passant, castling,
    and promotion.

    In general, you may not change the takeAction, isTerminal, getValue,
    getActions functions.
    """

    def __init__(self, Player=0, Board=None):
        """
        Initialize the chess game state.

        Input  : Player (int)         — 0 for White (MAX player), 1 for Black (MIN player).
                 Board (chess.Board | None) — position to use; None creates the
                 standard starting position.
        Output : None
        Example: ChessGame()             →  starting position, the MAX player (White) to move
                 ChessGame(1, my_board)  →  given position, the MIN player (Black) to move
        """
        State.__init__(self, Player)
        if Board is None:
            self.board = ch.Board()
        else:
            self.board = Board
        self.board.turn = (self.player == 0)

    def getActions(self):
        """
        Return all legal moves available to the current player.

        Input  : None
        Output : list of chess.Move — all pseudo-legal moves that leave the
                 king out of check are excluded; castling and en passant are
                 included when applicable.
        Example: getActions() on the starting position  →  list of 20 moves
        """
        return list(self.board.legal_moves)

    def takeAction(self, action):
        """
        Apply a move and return the resulting state.

        Input  : action (chess.Move) — a move returned by getActions().
        Output : ChessGame — new state after the move, with the other player
                 to move.
        Example: takeAction(chess.Move.from_uci('e2e4')) on the starting position
                 →  board with pawn advanced to e4, the MIN player to move
        """
        new_board = self.board.copy()
        new_board.push(action)
        return ChessGame(1 - self.player, new_board)

    def isTerminal(self):
        """
        Return True when the game has ended (checkmate, stalemate, or any
        other draw condition recognised by python-chess).

        Input  : None
        Output : bool
        Example: isTerminal() on the starting position        →  False
                 isTerminal() after a checkmate sequence      →  True
        """
        return self.board.is_game_over()

    def getValue(self):
        """
        Return the utility of this terminal state from the MAX player's perspective.

        Only call this when isTerminal() is True.
        Input  : None
        Output : float — +1.0 if the MAX player (White) won, -1.0 if the MIN player (Black)
                 won, 0.0 for a draw.
        Example: getValue() after White checkmates Black  →  1.0
                 getValue() after stalemate               →  0.0
        """
        if not self.isTerminal():
            raise ValueError("getValue() called on a non-terminal chess state.")
        outcome = self.board.outcome()
        result = outcome.result()
        if result == '1-0':
            return 1.0
        elif result == '0-1':
            return -1.0
        return 0.0

    def Evaluate(self):
        """
        Return a heuristic estimate of the state's value for the MAX player (White).

        *** THIS IS A STARTING IMPLEMENTATION. ***
        You are strongly encouraged to replace or extend it.
        The current version combines seven factors:
          - Material balance (pawn=1, knight/bishop=3, rook=5, queen=9)
          - Center control (attacks on d4/d5/e4/e5)
          - Pawn structure (doubled-pawn penalty, isolated-pawn penalty,
            pawn-chain support bonus)
          - King safety (friendly pawns adjacent to the king)
          - Piece tropism (pieces closer to the enemy king score higher)
          - Mobility (number of legal moves for the side to move)
          - Threats (value of opponent pieces attacked by friendly pieces)
          - Check bonus and loose-piece penalty

        The score is normalised so that the result lies approximately in [-1, 1].
        Positive values favour the MAX player (White); negative values favour the MIN player (Black).

        Ideas for improvement:
          - Use an endgame-specific evaluation (e.g. king activity matters more).
          - Add passed-pawn bonuses.
          - Use piece-square tables for positional bonuses.
          - You can also randomise among equally-scored actions in getBestMove()
            rather than always picking the first one found.

        Input  : None
        Output : float — getValue() if the state is terminal; otherwise a
                 normalised heuristic score where positive values favour the
                 MAX player (White) and negative values favour the MIN player (Black).
        Example: Evaluate() on the starting position                →  ~0.0  (roughly equal)
                 Evaluate() on a terminal state the MAX player won  →  1.0
        """
        if self.isTerminal():
            return self.getValue()
        board = self.board
        piece_values = {
            ch.PAWN: 1, ch.KNIGHT: 3, ch.BISHOP: 3,
            ch.ROOK: 5, ch.QUEEN: 9, ch.KING: 0
        }

        def total_score(color):
            material = sum(len(board.pieces(pt, color)) * val for pt, val in piece_values.items())
            center_squares = [ch.D4, ch.D5, ch.E4, ch.E5]
            center = sum(1 for sq in center_squares if board.is_attacked_by(color, sq))
            pawns = board.pieces(ch.PAWN, color)
            files = [ch.square_file(sq) for sq in pawns]
            structure = 0
            for sq in pawns:
                f = ch.square_file(sq)
                r = ch.square_rank(sq)
                if files.count(f) > 1:
                    structure -= 0.5
                if f - 1 not in files and f + 1 not in files:
                    structure -= 0.5
                for df in [-1, 1]:
                    neighbor_file = f + df
                    if 0 <= neighbor_file <= 7:
                        support_sq = ch.square(neighbor_file, r - 1) if color == ch.WHITE else ch.square(neighbor_file, r + 1)
                        if board.piece_at(support_sq) == ch.Piece(ch.PAWN, color):
                            structure += 0.3
                            break
            king_sq = board.king(color)
            safety = 0
            if king_sq:
                danger_zone = [sq for sq in ch.SQUARES if ch.square_distance(sq, king_sq) <= 1]
                safety += sum(1 for sq in danger_zone if board.piece_at(sq) and board.piece_at(sq).piece_type == ch.PAWN and board.piece_at(sq).color == color)
            tropism = 0
            enemy_king = board.king(not color)
            if enemy_king:
                for pt in [ch.QUEEN, ch.ROOK, ch.BISHOP, ch.KNIGHT]:
                    for sq in board.pieces(pt, color):
                        tropism += 1 / (1 + ch.square_distance(sq, enemy_king))
            mobility = len(list(board.legal_moves)) if board.turn == color else 0
            threats = 0
            for sq in ch.SQUARES:
                p = board.piece_at(sq)
                if p and p.color != color and board.is_attacked_by(color, sq):
                    threats += piece_values.get(p.piece_type, 0)
            checks_bonus = 0.5 if board.is_check() and board.turn != color else 0
            loose_penalty = 0
            for pt in [ch.QUEEN, ch.ROOK, ch.BISHOP, ch.KNIGHT]:
                for sq in board.pieces(pt, color):
                    if len(board.attackers(not color, sq)) > len(board.attackers(color, sq)):
                        loose_penalty += piece_values[pt] * 0.2
            return (
                4.5 * material + 0.2 * center + 0.2 * structure +
                0.3 * safety + 0.2 * tropism + 0.2 * mobility +
                0.1 * threats + 2 * checks_bonus - 1.0 * loose_penalty
            )

        white_score = total_score(ch.WHITE)
        black_score = total_score(ch.BLACK)
        total = white_score + black_score
        if total == 0:
            return 0.0
        return -(black_score - white_score) / total

    def getBestMove(self, depth=3, algo="default"):
        """
        Return the best move found by the selected search algorithm.

        *** YOU ARE ENCOURAGED TO MODIFY THIS FUNCTION. ***
        You can change the default depth, switch algorithms, or replace the
        search entirely with your own approach (e.g. a neural-network policy,
        a Monte Carlo rollout, or a random baseline).

        Input  : depth (int)  — search depth (minimax plies) or MCTS iteration
                 count, depending on the algorithm chosen.
                 algo (str)   — 'default' uses minimax with alpha-beta pruning
                 and Evaluate(); 'MCTS' uses Monte Carlo Tree Search with
                 'depth' playouts.
        Output : chess.Move | None — the recommended move, or None if the
                 position is terminal.
        Example: getBestMove(depth=3)                →  best move found in 3-ply search
                 getBestMove(depth=200, algo='MCTS') →  best move after 200 playouts
        """
        if self.isTerminal():
            return None
        alpha = -float('inf')
        beta = float('inf')
        if algo == "default":
            return minimaxAlphaBetaRecursiveWithEvaluationFunction(self, alpha, beta, 0, depth)[1]
        elif algo == "MCTS":
            return MCTS_best_action(self, iterations=depth)
        return None

    def __repr__(self):
        """
        Return a human-readable string of the board using python-chess notation.

        Input  : None
        Output : str — ASCII representation of the board.
        Example: repr(ChessGame())  →  standard starting-position ASCII board
        """
        return str(self.board)
