from typing import Dict
from typing import List


class FiniteAutomata:
    states: List[int]
    symbols: List[str]
    final: List[int]
    start: List[int]
    transitions: Dict[int, List[int]]


def create_random_fsm(states, alphabet, final_states):
    transitions = {state: {} for state in states}
