import math
import time


class SearchTimeout(Exception):
    pass


class CustomAgent:
    def __init__(self):
        self.depth_limit = 3
        self.time_limit = 0.75
        self.deadline = 0
        self.table = {}

    def takeAction(self, state):
        actions = state.getActions()
        if not actions:
            return None

        self.deadline = time.perf_counter() + self.time_limit
        self.table = {}

        actions = self._order_actions(state, actions)
        best_action = actions[0]

        try:
            if state.player == 0:
                best_value = -math.inf
                alpha = -math.inf
                beta = math.inf

                for action in actions:
                    self._check_time()

                    child = state.takeAction(action)
                    value = self._alphabeta(child, self.depth_limit - 1, alpha, beta)

                    if value > best_value:
                        best_value = value
                        best_action = action

                    alpha = max(alpha, best_value)

                    if alpha >= beta:
                        break

            else:
                best_value = math.inf
                alpha = -math.inf
                beta = math.inf

                for action in actions:
                    self._check_time()

                    child = state.takeAction(action)
                    value = self._alphabeta(child, self.depth_limit - 1, alpha, beta)

                    if value < best_value:
                        best_value = value
                        best_action = action

                    beta = min(beta, best_value)

                    if alpha >= beta:
                        break

        except SearchTimeout:
            pass

        return best_action

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout()

    def _alphabeta(self, state, depth, alpha, beta):
        self._check_time()

        if state.isTerminal():
            return 10000.0 * state.getValue()

        if depth <= 0:
            return self._evaluate(state)

        key = self._state_key(state, depth)
        if key in self.table:
            return self.table[key]

        actions = state.getActions()
        if not actions:
            return self._evaluate(state)

        actions = self._order_actions(state, actions)
        completed_search = True

        if state.player == 0:
            value = -math.inf

            for action in actions:
                child = state.takeAction(action)
                value = max(value, self._alphabeta(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)

                if alpha >= beta:
                    completed_search = False
                    break

        else:
            value = math.inf

            for action in actions:
                child = state.takeAction(action)
                value = min(value, self._alphabeta(child, depth - 1, alpha, beta))
                beta = min(beta, value)

                if alpha >= beta:
                    completed_search = False
                    break

        if completed_search:
            self.table[key] = value

        return value

    def _evaluate(self, state):
        if state.isTerminal():
            return 10000.0 * state.getValue()

        red_score = 0
        black_score = 0

        board = state.board

        for r in range(8):
            for c in range(8):
                piece = board[r][c]

                if piece == "r":
                    red_score += 100
                    red_score += (7 - r) * 5

                elif piece == "R":
                    red_score += 175

                elif piece == "b":
                    black_score += 100
                    black_score += r * 5

                elif piece == "B":
                    black_score += 175

        return red_score - black_score

    def _order_actions(self, state, actions):
        scored_actions = []

        for action in actions:
            score = 0

            jump_count = 0
            for i in range(1, len(action)):
                if abs(action[i][0] - action[i - 1][0]) == 2:
                    jump_count += 1

            score += 1000 * jump_count

            start_r, start_c = action[0]
            end_r, end_c = action[-1]
            piece = state.board[start_r][start_c]

            if piece == "r" and end_r == 0:
                score += 300
            elif piece == "b" and end_r == 7:
                score += 300

            if piece == "r":
                score += 7 - end_r
            elif piece == "b":
                score += end_r

            scored_actions.append((score, action))

        scored_actions.sort(key=lambda item: item[0], reverse=True)
        return [action for score, action in scored_actions]

    def _state_key(self, state, depth):
        board_key = tuple(tuple(row) for row in state.board)

        return (
            board_key,
            state.player,
            depth,
            getattr(state, "no_progress_count", 0),
        )
