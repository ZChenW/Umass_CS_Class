import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from state import State


class checkers(State):
    """
    Checkers (8×8 draughts) game state.

    The MAX player controls red pieces ('r' / 'R' kings); the MIN player controls
    black pieces ('b' / 'B' kings).  Red moves toward row 0; black moves toward
    row 7.  Jumps are mandatory when available.  The game ends when the
    current player has no legal moves or after 30 consecutive moves without a
    capture or promotion (draw).

    In general, you may not change the takeAction, isTerminal, getValue,
    getActions functions.
    """

    def __init__(self, Player=0, Board=None, no_progress_count=0):
        """
        Initialize the checkers game state.

        Input  : Player (int)            — 0 for red (MAX player), 1 for black (MIN player).
                 Board (list[list[str]] | None) — 8×8 board; None creates the
                 standard starting position.
                 no_progress_count (int) — consecutive moves without capture or
                 promotion; used for the 30-move draw rule.
        Output : None
        Example: checkers()          →  standard starting position, the MAX player to move
                 checkers(1, board)  →  given board, the MIN player to move
        """
        State.__init__(self, Player)
        if Board is None:
            self.board = [
                [
                    '.' if (i + j) % 2 == 0 else
                    ('b' if i < 3 else ('r' if i > 4 else ' '))
                    for j in range(8)
                ]
                for i in range(8)
            ]
        else:
            self.board = Board
        self.no_progress_count = no_progress_count

    def getActions(self):
        """
        Return all legal moves for the current player.

        Jumps are mandatory: if any jump is available the list contains only
        jump sequences; otherwise normal single-step moves are returned.
        Each move is a list of (row, col) waypoints starting at the piece's
        current position.

        Input  : None
        Output : list of list of (int, int) — each element is an ordered path
                 of board positions; a two-element list is a simple step, a
                 longer list is a multi-jump sequence.
        Example: getActions() at the start of the game
                 →  list of 7 single-step moves for red's front pieces
        """
        all_jumps = []
        normal_moves = []
        player = 'r' if self.player == 0 else 'b'
        player_pieces = {'r', 'R'} if player == 'r' else {'b', 'B'}

        def find_normal_moves(row, col, player):
            moves = []
            piece = self.board[row][col]
            if piece.isupper():
                directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
            else:
                if piece == 'r':
                    directions = [(-1, -1), (-1, 1)]
                elif piece == 'b':
                    directions = [(1, -1), (1, 1)]
                else:
                    directions = []
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == ' ':
                    moves.append([(row, col), (r, c)])
            return moves

        def find_jumps(board, pos, player, path=None, captured=None):
            if path is None:
                path = [pos]
            if captured is None:
                captured = set()
            jumps = []
            row, col = pos
            opponent_pieces = {'r', 'R'} if player == 'b' else {'b', 'B'}
            piece = board[row][col]
            if piece.isupper():
                directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
            else:
                if piece == 'r':
                    directions = [(-1, -1), (-1, 1)]
                elif piece == 'b':
                    directions = [(1, -1), (1, 1)]
                else:
                    directions = []
            found_jump = False
            for dr, dc in directions:
                mid_r, mid_c = row + dr, col + dc
                end_r, end_c = row + 2 * dr, col + 2 * dc
                if 0 <= mid_r < 8 and 0 <= mid_c < 8 and 0 <= end_r < 8 and 0 <= end_c < 8:
                    if board[mid_r][mid_c] in opponent_pieces and board[end_r][end_c] == ' ':
                        if (mid_r, mid_c) not in captured:
                            new_board = [r.copy() for r in board]
                            new_board[end_r][end_c] = new_board[row][col]
                            new_board[row][col] = ' '
                            new_board[mid_r][mid_c] = ' '
                            just_crowned = False
                            if new_board[end_r][end_c] == 'r' and end_r == 0:
                                new_board[end_r][end_c] = 'R'
                                just_crowned = True
                            elif new_board[end_r][end_c] == 'b' and end_r == 7:
                                new_board[end_r][end_c] = 'B'
                                just_crowned = True
                            new_captured = captured.copy()
                            new_captured.add((mid_r, mid_c))
                            new_path = path + [(end_r, end_c)]
                            # In American checkers, crowning ends the turn immediately.
                            if just_crowned:
                                jumps.append(new_path)
                            else:
                                further_jumps = find_jumps(new_board, (end_r, end_c), player, new_path, new_captured)
                                if further_jumps:
                                    jumps.extend(further_jumps)
                                else:
                                    if len(new_path) > 1:
                                        jumps.append(new_path)
                            found_jump = True
            if not found_jump:
                return [path] if len(path) > 1 else []
            return jumps

        for i in range(8):
            for j in range(8):
                if self.board[i][j] in player_pieces:
                    jumps = find_jumps(self.board, (i, j), player)
                    if jumps:
                        all_jumps.extend(jumps)
                    else:
                        normal_moves.extend(find_normal_moves(i, j, player))

        return all_jumps if all_jumps else normal_moves

    def takeAction(self, action):
        """
        Apply a move and return the resulting state.

        Input  : action (list of (int, int)) — ordered path of board positions.
                 The first element is the piece's starting square; each subsequent
                 element is the next landing square.  Single-step moves have
                 length 2; multi-jumps are longer.
        Output : checkers — new state after the move, with the other player to move.
                 Captured pieces are removed; pieces are promoted to kings when
                 they reach the far row.  no_progress_count resets on capture or
                 promotion and increments otherwise.
        Example: takeAction([[(5,0),(4,1)]]) on the starting board
                 →  red piece moved from (5,0) to (4,1), the MIN player to move
        """
        newboard = copy.deepcopy(self.board)
        no_progress = self.no_progress_count
        start_r, start_c = action[0]
        piece = newboard[start_r][start_c]
        captured = False
        promoted = False
        for i in range(1, len(action)):
            newboard[start_r][start_c] = ' '
            end_r, end_c = action[i]
            if abs(end_r - start_r) == 2:
                mid_r = (start_r + end_r) // 2
                mid_c = (start_c + end_c) // 2
                newboard[mid_r][mid_c] = ' '
                captured = True
            newboard[end_r][end_c] = piece
            if piece == 'r' and end_r == 0:
                piece = 'R'
                newboard[end_r][end_c] = piece
                promoted = True
            elif piece == 'b' and end_r == 7:
                piece = 'B'
                newboard[end_r][end_c] = piece
                promoted = True
            start_r, start_c = end_r, end_c
        no_progress = 0 if captured or promoted else no_progress + 1
        next_player = 1 if self.player == 0 else 0
        return checkers(next_player, newboard, no_progress)

    def isTerminal(self):
        """
        Return True when the game has ended.

        The game ends when the current player has no legal moves (all their
        pieces are captured or completely blocked) or after 30 consecutive
        moves without a capture or promotion (draw).

        Input  : None
        Output : bool
        Example: isTerminal() on the starting board      →  False
                 isTerminal() when red has no pieces     →  True
        """
        return not self.getActions() or self.no_progress_count >= 30

    def getValue(self):
        """
        Return the utility of this terminal state from the MAX player's perspective.

        Only call this when isTerminal() is True.
        Input  : None
        Output : int — +1 if the MAX player (red) wins, -1 if the MIN player (black) wins,
                 0 for a draw (30-move rule).
        Example: getValue() when black has no moves        →  +1  (red wins)
                 getValue() after 30 no-progress moves     →  0   (draw)
        """
        if not self.isTerminal():
            raise ValueError("getValue() called on a non-terminal checkers state.")
        if self.no_progress_count >= 30:
            return 0
        if not self.getActions():
            return 1 if (1 - self.player) == 0 else -1
        return 0

    def _is_threatened(self, row, col):
        """
        Return True if the piece at (row, col) can be captured by the opponent
        on the opponent's next move.

        Input  : row (int), col (int) — square to check.
        Output : bool
        Example: _is_threatened(5, 0) when an opponent piece is at (4, 1) and
                 (6, -1) is off-board  →  False (no landing square behind)
        """
        piece = self.board[row][col]
        if piece == ' ' or piece == '.':
            return False
        is_red = piece.lower() == 'r'
        opponent_pieces = {'b', 'B'} if is_red else {'r', 'R'}
        for dr in [-1, 1]:
            for dc in [-1, 1]:
                mid_r, mid_c = row + dr, col + dc
                land_r, land_c = row + 2 * dr, col + 2 * dc
                if not (0 <= mid_r < 8 and 0 <= mid_c < 8):
                    continue
                if not (0 <= land_r < 8 and 0 <= land_c < 8):
                    continue
                attacker = self.board[mid_r][mid_c]
                if attacker not in opponent_pieces:
                    continue
                if attacker.islower():
                    if attacker == 'b' and dr != 1:
                        continue
                    if attacker == 'r' and dr != -1:
                        continue
                if self.board[land_r][land_c] == ' ':
                    return True
        return False

    def Evaluate(self):
        """
        Return a heuristic estimate of the state's value for the MAX player (red).

        *** THIS IS A STARTING IMPLEMENTATION. ***
        You are strongly encouraged to replace or extend it.
        The current version considers only four board parameters:
          r_pieces     — number of red pieces (kings count as 1)
          b_pieces     — number of black pieces (kings count as 1)
          r_threatened — number of red pieces currently threatened by a jump
          b_threatened — number of black pieces currently threatened by a jump

        Positive values favour the MAX player (red); negative values favour the MIN player (black).

        Ideas for a better heuristic:
          - Weight kings more heavily than regular pieces.
          - Include positional bonuses (center control, back-row defence).
          - Count mobility (number of available moves).
          - You can also randomise among equally-scored actions in getBestMove()
            rather than always picking the first one found.

        Input  : None
        Output : float — getValue() if the state is terminal; otherwise a
                 heuristic score normalised to [-1, 1] where positive values
                 favour the MAX player (red) and negative values favour the
                 MIN player (black).
        Example: Evaluate() on the starting board                    →  0.0  (equal piece counts,
                 equal threats)
                 Evaluate() on a terminal state the MAX player won   →  1
        """
        if self.isTerminal():
            return self.getValue()
        r_pieces = b_pieces = r_threatened = b_threatened = 0
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece.lower() == 'r':
                    r_pieces += 1
                    if self._is_threatened(i, j):
                        r_threatened += 1
                elif piece.lower() == 'b':
                    b_pieces += 1
                    if self._is_threatened(i, j):
                        b_threatened += 1
        score = (r_pieces - b_pieces) - (r_threatened - b_threatened)
        denom = r_pieces + b_pieces
        if denom == 0:
            return 0.0
        return score / denom

    def getBestMove(self):
        """
        Return a greedy best action using the current Evaluate() function.

        The MAX player picks the action that maximises Evaluate() on the successor state;
        the MIN player picks the action that minimises it.  Ties are broken by the order
        in which getActions() returns moves (deterministic).

        *** YOU ARE ENCOURAGED TO MODIFY THIS FUNCTION. ***
        For example, you could:
          - Sample actions with probability proportional to their Evaluate() scores.
          - Use a deeper look-ahead (minimax or alpha-beta search).
          - Combine Evaluate() with getValue() when the successor is terminal.

        Input  : None
        Output : list of (int, int) | None — the chosen move path,
                 or None if the state is terminal.
        Example: getBestMove() on the starting board
                 →  some legal move for red  (all score equally with the default
                    heuristic, so the first move returned by getActions() is chosen)
                 getBestMove() on a terminal board  →  None
        """
        if self.isTerminal():
            return None
        best_action = None
        best_value = -float('inf') if self.player == 0 else float('inf')
        for action in self.getActions():
            value = self.takeAction(action).Evaluate()
            if self.player == 0 and value > best_value:
                best_value = value
                best_action = action
            elif self.player == 1 and value < best_value:
                best_value = value
                best_action = action
        return best_action

    def __repr__(self):
        """
        Return a human-readable string of the board.

        Input  : None
        Output : str — labelled 8×8 grid with column/row headers.
        Example: repr(checkers()) →  grid showing the standard starting position
        """
        result = "   " + " ".join(str(j) for j in range(8)) + "\n"
        result += "  +" + "--" * 8 + "+\n"
        for i in range(8):
            row_str = str(i) + " |"
            for j in range(8):
                row_str += self.board[i][j] + " "
            row_str += "|\n"
            result += row_str
        result += "  +" + "--" * 8 + "+\n"
        return result
