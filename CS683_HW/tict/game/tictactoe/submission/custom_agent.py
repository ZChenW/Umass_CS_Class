import random


class CustomAgent:
    """
    Example random agent. Replace takeAction() with your own logic.

    Your agent will be instantiated once and reused across games.
    """

    def __init__(self):
        self.cache = {}

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

        best_action = actions[0]
        if state.player == 0:
            best_value = -float("inf")
            alpha = -float("inf")
            beta = float("inf")
            for action in actions:
                value = self.alphabeta(state.takeAction(action), alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
        else:
            best_value = float("inf")
            alpha = -float("inf")
            beta = float("inf")
            for action in actions:
                value = self.alphabeta(state.takeAction(action), alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
        return best_action

    def alphabeta(self, state, alpha, beta):
        key = self.key(state)
        if key in self.cache:
            return self.cache[key]
        if state.isTerminal():
            value = state.getValue()
            self.cache[key] = value
            return value
        actions = state.getActions()
        if state.player == 0:
            value = -float("inf")
            for action in actions:
                value = max(
                    value, self.alphabeta(state.takeAction(action), alpha, beta)
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for action in actions:
                value = min(
                    value, self.alphabeta(state.takeAction(action), alpha, beta)
                )
                beta = min(beta, value)
                if alpha >= beta:
                    break
        self.cache[key] = value
        return value

    def key(self, state):
        board = state.board
        return (
            state.player,
            board[0][0],
            board[0][1],
            board[0][2],
            board[1][0],
            board[1][1],
            board[1][2],
            board[2][0],
            board[2][1],
            board[2][2],
        )
