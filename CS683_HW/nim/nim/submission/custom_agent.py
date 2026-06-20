import random


class CustomAgent:
    """
    Example random agent. Replace takeAction() with your own logic.

    Your agent will be instantiated once and reused across games.
    """

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
        piles = list(state.sticks)
        winning_action = self.misere_nim_action(piles)
        if winning_action in actions:
            return winning_action
        return actions[0]

    def misere_nim_action(self, piles):
        nonzero = [(i, pile) for i, pile in enumerate(piles) if pile > 0]
        if not nonzero:
            return None

        big_piles = [(i, pile) for i, pile in nonzero if pile > 1]
        ones_count = sum(1 for _, pile in nonzero if pile == 1)

        if not big_piles:
            return (1, nonzero[0][0])

        if len(big_piles) == 1:
            pile_index, pile_size = big_piles[0]
            if ones_count % 2 == 0:
                amount = pile_size - 1
            else:
                amount = pile_size
            if amount <= 0:
                amount = 1
            return (amount, pile_index)

        nim_sum = 0
        for pile in piles:
            nim_sum ^= pile

        if nim_sum == 0:
            for i, pile in nonzero:
                return (1, i)

        for i, pile in nonzero:
            target = pile ^ nim_sum
            if target < pile:
                amount = pile - target
                return (amount, i)

        return None
