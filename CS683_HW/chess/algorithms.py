import math
import random


# ---------------------------------------------------------------------------
# Minimax (exact, no pruning)
# ---------------------------------------------------------------------------

def minimax(state):
    """
    Run full minimax search from state and return the optimal value and action.

    Explores the entire game tree without pruning; guaranteed optimal but
    exponential in tree depth.
    Input  : state (State) — any non-terminal or terminal game state.
    Output : [value (numeric), best_action] — value is from Player 0's perspective;
             best_action is None at terminal nodes.
    Example: minimax(TicTacToe())  will output  [0, (1,1)] 
    corresponding to a draw (value 0) and the center position as the optimal first move.
    """
    best_action = None
    if state.isTerminal():
        return [state.getValue(), best_action]
    if state.player == 0:
        current_val = -math.inf
        for action in state.getActions():
            action_value = minimax(state.takeAction(action))[0]
            if action_value > current_val:
                current_val = action_value
                best_action = action
        return [current_val, best_action]
    else:
        current_val = math.inf
        for action in state.getActions():
            action_value = minimax(state.takeAction(action))[0]
            if action_value < current_val:
                current_val = action_value
                best_action = action
        return [current_val, best_action]


# ---------------------------------------------------------------------------
# Alpha-beta pruning (exact, full depth)
# ---------------------------------------------------------------------------

def minimaxAlphaBetaRecursive(state, alpha, beta):
    """
    Recursively run alpha-beta pruning from state with the given alpha/beta window.

    Prunes branches that cannot influence the result, reducing average-case cost.
    Input  : state (State), alpha (numeric) — best guaranteed value for Player 0 so far,
             beta (numeric) — best guaranteed value for Player 1 so far.
    Output : [value (numeric), best_action] — same semantics as minimax.
    """
    best_action = None
    if state.isTerminal():
        return [state.getValue(), best_action]
    if state.player == 0:
        current_val = -math.inf
        for action in state.getActions():
            action_val = minimaxAlphaBetaRecursive(state.takeAction(action), alpha, beta)[0]
            if action_val > current_val:
                current_val = action_val
                best_action = action
            alpha = max(alpha, action_val)
            if beta <= alpha:
                break
        return [current_val, best_action]
    else:
        current_val = math.inf
        for action in state.getActions():
            action_val = minimaxAlphaBetaRecursive(state.takeAction(action), alpha, beta)[0]
            if action_val < current_val:
                current_val = action_val
                best_action = action
            beta = min(beta, action_val)
            if beta <= alpha:
                break
        return [current_val, best_action]


def minimaxAlphaBeta(state):
    """
    Run alpha-beta pruning from state with an unbounded initial window.

    Convenience entry point that calls minimaxAlphaBetaRecursive with [-inf, inf].
    Input  : state (State) — the root game state to search from.
    Output : [value (numeric), best_action] — optimal value and the action that achieves it.
    Example: minimaxAlphaBeta(TicTacToe())  →  [0, (1,1)]
    """
    return minimaxAlphaBetaRecursive(state, -math.inf, math.inf)


def minimaxAlphaBetaRecursiveWithEvaluationFunction(state, alpha, beta, current_depth, max_depth):
    """
    Run depth-limited alpha-beta pruning, falling back to Evaluate() at the depth limit.

    Cuts the search off at max_depth and uses state.Evaluate() as a heuristic
    instead of searching further, making deep game trees tractable.
    Input  : state (State), alpha (numeric), beta (numeric),
             current_depth (int) — depth of this node from the root (start at 0),
             max_depth (int) — maximum depth before calling Evaluate().
    Output : [value (numeric), best_action] — heuristic-backed value and best action found.
    Example: minimaxAlphaBetaRecursiveWithEvaluationFunction(checkers(), -inf, inf, 0, 4)
             →  [0.62, ((2,1),(3,2))]  (best checkers move at depth 4)
    """
    best_action = None
    if state.isTerminal():
        return [state.getValue(), best_action]
    if current_depth >= max_depth:
        return [state.Evaluate(), best_action]
    if state.player == 0:
        current_val = -math.inf
        for action in state.getActions():
            action_val = minimaxAlphaBetaRecursiveWithEvaluationFunction(
                state.takeAction(action), alpha, beta, current_depth + 1, max_depth)[0]
            if action_val > current_val:
                current_val = action_val
                best_action = action
            alpha = max(alpha, action_val)
            if beta <= alpha:
                break
        return [current_val, best_action]
    else:
        current_val = math.inf
        for action in state.getActions():
            action_val = minimaxAlphaBetaRecursiveWithEvaluationFunction(
                state.takeAction(action), alpha, beta, current_depth + 1, max_depth)[0]
            if action_val < current_val:
                current_val = action_val
                best_action = action
            beta = min(beta, action_val)
            if beta <= alpha:
                break
        return [current_val, best_action]


# ---------------------------------------------------------------------------
# Monte Carlo Tree Search (MCTS)
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    A single node in the MCTS search tree, corresponding to one game state.

    Stores visit counts, accumulated wins, and the list of actions not yet expanded,
    enabling UCB-guided selection and incremental tree growth.
    """

    def __init__(self, state, parent=None, action=None):
        """
        Initialize an MCTS node for the given state.

        Input  : state (State) — the game state this node represents;
                 parent (MCTSNode | None) — the parent node, or None for the root;
                 action — the action taken from the parent to reach this node.
        Output : None
        Example: root = MCTSNode(TicTacToe())  →  unvisited root node
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = state.getActions()

    def is_fully_expanded(self):
        """
        Return True if every legal action from this node has been expanded into a child.

        Input  : None
        Output : bool — True when untried_actions is empty.
        Example: node.is_fully_expanded()  →  False if there are still unvisited moves
        """
        return len(self.untried_actions) == 0

    def compute_ucb(self, child, C):
        """
        Compute the UCB1 score for child relative to this node.

        Balances exploitation (win rate) with exploration (visit frequency).
        Input  : child (MCTSNode) — a child of this node;
                 C (float) — exploration constant (higher values encourage exploration).
        Output : float — UCB1 score; infinity if child has never been visited.
        """
        if child.visits == 0:
            return float('inf')
        eps = 1e-10
        exploitation = child.wins / child.visits
        exploration = C * ((math.log(self.visits + eps) / child.visits) ** 0.5)
        return exploitation + exploration

    def best_child(self, C=1.414):
        """
        Return the child with the highest UCB1 score under exploration constant C.

        Use C=0 to select greedily by win rate alone (used when extracting the final move).
        Input  : C (float) — exploration constant; default 1.414 (≈ sqrt(2)).
        Output : MCTSNode — the child node with the maximum UCB1 score.
        Example: node.best_child(C=0)  →  the most-visited, highest-win-rate child
        """
        best = self.children[0]
        best_score = self.compute_ucb(best, C)
        for child in self.children[1:]:
            score = self.compute_ucb(child, C)
            if score > best_score:
                best = child
                best_score = score
        return best

    def expand(self):
        """
        Add one new child node by trying the next untried action.

        Input  : None
        Output : MCTSNode — the newly created child node.
        Example: child = node.expand()  →  new child representing an unexplored move
        """
        action = self.untried_actions.pop()
        next_state = self.state.takeAction(action)
        child_node = MCTSNode(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        """
        Propagate the simulation result up through all ancestor nodes.

        Input  : result (numeric) — the outcome of the simulation (e.g. 1 win, -1 loss, 0 draw).
        Output : None (modifies visits and wins in-place up the tree).
        Example: node.backpropagate(1)  →  increments visits and wins for node and all ancestors
        """
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def default_policy(self):
        """
        Simulate a random playout from this node's state and return the terminal value.

        Stops early (returning 0) if the playout exceeds 100 steps without terminating.
        Input  : None
        Output : numeric — getValue() of the terminal state, or 0 if the cap is reached.
        Example: node.default_policy()  →  1  (random play reached a Player-0 win)
        """
        current_state = self.state
        count = 0
        while not current_state.isTerminal() and count < 100:
            count += 1
            actions = current_state.getActions()
            current_state = current_state.takeAction(random.choice(actions))
        return current_state.getValue() if count < 100 else 0

    def heuristic_policy_soft_greedy(self, p=0.9, delta=0.2):
        """
        Simulate a playout using a soft-greedy heuristic guided by Evaluate().

        At each step, takes the best action (by Evaluate) with probability p, and the
        second-best (if within delta of the best) with probability 1-p.
        Input  : p (float) — probability of choosing the best action (default 0.9);
                 delta (float) — maximum value gap to consider the second-best (default 0.2).
        Output : int — 1 if Player 0 wins, -1 if Player 1 wins, 0 for draw or cap reached.
        Example: node.heuristic_policy_soft_greedy(p=0.9)  →  1  (Player 0 likely wins)
        """
        current_state = self.state
        count = 0
        while not current_state.isTerminal() and count < 100:
            count += 1
            actions = current_state.getActions()
            evaluated = []
            for action in actions:
                next_state = current_state.takeAction(action)
                value = next_state.Evaluate()
                evaluated.append((value, action))
            reverse = current_state.player == 0
            evaluated.sort(key=lambda x: x[0], reverse=reverse)
            best_value, best_action = evaluated[0]
            second_best_action = None
            if len(evaluated) > 1:
                second_value, second_action = evaluated[1]
                if abs(best_value - second_value) <= delta:
                    second_best_action = second_action
            if random.random() < p or second_best_action is None:
                chosen_action = best_action
            else:
                chosen_action = second_best_action
            current_state = current_state.takeAction(chosen_action)
        if current_state.isTerminal():
            return current_state.getValue()
        eval_score = current_state.Evaluate()
        if eval_score > 0:
            return 1
        elif eval_score < 0:
            return -1
        return 0


def mcts_search(root, iterations=100):
    """
    Run MCTS for a fixed number of iterations starting from root and return the best child.

    Each iteration selects a node to expand (or expands the root if not fully expanded),
    simulates a playout, and backpropagates the result.
    Input  : root (MCTSNode) — the root node of the search tree;
             iterations (int) — number of simulation iterations (default 100).
    Output : MCTSNode — the best child of root after all iterations (greedy by win rate).
    Example: mcts_search(MCTSNode(TicTacToe()), iterations=500)  →  child node for best move
    """
    for _ in range(iterations):
        node = root
        if not node.is_fully_expanded():
            node = node.expand()
        else:
            node = node.best_child()
        result = node.heuristic_policy_soft_greedy()
        if result == 1 and root.state.player == 0:
            result1 = 1
        elif result == -1 and root.state.player == 0:
            result1 = -1
        elif result == -1 and root.state.player == 1:
            result1 = 1
        elif result == 1 and root.state.player == 1:
            result1 = -1
        else:
            result1 = 0
        node.backpropagate(result1)
    return root.best_child(C=0)


def MCTS_best_action(root, iterations=100):
    """
    Return the best action from root's state using Monte Carlo Tree Search.

    Wraps root in an MCTSNode, runs mcts_search, and extracts the action of the best child.
    Input  : root (State) — the game state to search from;
             iterations (int) — number of MCTS simulations (default 100).
    Output : action — the action leading to the best child node found by MCTS.
    Example: MCTS_best_action(TicTacToe(), iterations=200)  →  (1, 1)  (center square)
    """
    root = MCTSNode(root)
    best = mcts_search(root, iterations)
    return best.action
