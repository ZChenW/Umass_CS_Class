import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from state import State


class TicTacToe(State):
    """
    Tic-Tac-Toe game state.

    The MAX player places 'X'; the MIN player places 'O'.
    The board is a 3×3 list of lists; empty squares are ' '.
    """

    def __init__(self, Player=0, Board=None):
        """
        Initialize the Tic-Tac-Toe state.

        Input  : Player (int) — 0 for X (MAX player), 1 for O (MIN player).
                 Board (list[list[str]] | None) — 3×3 board; None creates an empty board.
        Output : None
        Example: TicTacToe()            →  empty board, the MAX player to move
                 TicTacToe(1, my_board) →  given board, the MIN player to move
        """
        State.__init__(self, Player)
        if Board is None:
            self.board = [[' ' for _ in range(3)] for _ in range(3)]
        else:
            self.board = Board

    def getActions(self):
        """
        Return all empty squares as (row, col) pairs.

        Input  : None
        Output : list of (int, int) — every (i, j) where self.board[i][j] == ' '.
        Example: getActions() on an empty board →  [(0,0),(0,1),...,(2,2)]  (9 actions)
                 getActions() on a full board   →  []
        """
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']

    def takeAction(self, action):
        """
        Place the current player's mark at action and return the resulting state.

        Input  : action (int, int) — (row, col) of the square to mark.
        Output : TicTacToe — new state with the mark placed and the other player to move.
        Example: takeAction((1, 1)) on an empty board, the MAX player
                 →  board with 'X' at center, the MIN player to move
        """
        newboard = copy.deepcopy(self.board)
        mark = 'X' if self.player == 0 else 'O'
        if newboard[action[0]][action[1]] == ' ':
            newboard[action[0]][action[1]] = mark
        return TicTacToe(1 - self.player, newboard)

    def check_winner(self, thePlayer):
        """
        Return True if thePlayer has three marks in a row, column, or diagonal.

        Input  : thePlayer (int) — 0 checks for 'X', 1 checks for 'O'.
        Output : bool
        Example: check_winner(0) on a board where 'X' fills the top row →  True
        """
        mark = 'X' if thePlayer == 0 else 'O'
        for i in range(3):
            if all(self.board[i][j] == mark for j in range(3)):
                return True
            if all(self.board[j][i] == mark for j in range(3)):
                return True
        if all(self.board[i][i] == mark for i in range(3)):
            return True
        if all(self.board[i][2 - i] == mark for i in range(3)):
            return True
        return False

    def isTerminal(self):
        """
        Return True if the game has ended (a player has won or the board is full).

        Input  : None
        Output : bool
        Example: isTerminal() on an empty board                    →  False
                 isTerminal() after the MAX player fills a row     →  True
        """
        return self.check_winner(0) or self.check_winner(1) or not self.getActions()

    def getValue(self):
        """
        Return the utility of this terminal state from the MAX player's perspective.

        Only call this when isTerminal() is True.
        Input  : None
        Output : int — +1 if the MAX player (X) won, -1 if the MIN player (O) won,
                 0 for a draw.
        Example: getValue() when X has three in a row     →  1
                 getValue() on a full board with no winner →  0
        """
        if not self.isTerminal():
            raise ValueError("getValue() called on a non-terminal Tic-Tac-Toe state.")
        if self.check_winner(0):
            return 1
        if self.check_winner(1):
            return -1
        return 0

    def Evaluate(self):
        """
        Return a heuristic estimate of the state's value for the MAX player.

        *** THIS IS A STARTING IMPLEMENTATION. ***
        You are strongly encouraged to replace or extend it.
        The current version always returns 0, giving no guidance to the search.

        Positive values favor the MAX player (X); negative values favor the MIN player (O).

        Ideas for a better heuristic:
          - Count lines (rows, cols, diagonals) that only contain X marks (+score)
            and lines that only contain O marks (-score).
          - Weight two-in-a-row threats more heavily than single marks.
          - You can also randomise among equally-scored actions in getBestMove()
            rather than always picking the first one found.

        Input  : None
        Output : float — getValue() if the state is terminal; otherwise a
                 heuristic score where positive values favour the MAX player
                 and negative values favour the MIN player.
        Example: Evaluate() on a terminal state where X won  →  1
                 Evaluate() on any non-terminal board        →  0.0  (with this starter code)
        """
        if self.isTerminal():
            return self.getValue()
        return 0

    def getBestMove(self):
        """
        Return a greedy best action using the current Evaluate() function.

        The MAX player picks the action that maximises Evaluate() on the successor state;
        the MIN player picks the action that minimises it.  Ties are broken by the order
        in which getActions() returns moves (deterministic).

        *** YOU ARE ENCOURAGED TO MODIFY THIS FUNCTION. ***
        For example, you could:
          - Sample actions with probability proportional to their Evaluate() scores.
          - Use a deeper look-ahead (call getBestMove recursively or use minimax).
          - Combine Evaluate() with getValue() when the successor is terminal.

        Input  : None
        Output : (int, int) | None — the chosen (row, col) action,
                 or None if the state is terminal.
        Example: getBestMove() on an empty board →  (0, 0)  (first action, all scores tie at 0)
                 getBestMove() on a terminal board →  None
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
        Output : str — 3-row grid with separators.
        Example: repr(TicTacToe()) →  ' |   |   |\n---------\n ...'
        """
        s = ""
        for row in range(3):
            s += " | ".join(self.board[row]) + " |\n"
            if row < 2:
                s += "---------\n"
        return s
