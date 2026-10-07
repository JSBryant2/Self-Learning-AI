"""One agent. Version 1 chooses randomly and does not learn."""

import random


def create_agent():
    return {
        "available_actions": ["move_forward", "turn_left", "turn_right", "interact"],
        "current_observation": None,
        "last_action": None,
    }


def choose_action(agent, observation):
    # Store what the agent receives. Random choice does not use its contents yet.
    agent["current_observation"] = observation
    action = random.choice(agent["available_actions"])
    agent["last_action"] = action
    return action
