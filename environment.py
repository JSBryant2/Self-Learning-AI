"""The physical world and its rules. No decisions are made here."""

DIRECTIONS = ["north", "east", "south", "west"]
FORWARD_OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def create_environment():
    # Rows run top to bottom; columns run left to right. # is wall, . is floor.
    layout = [
        "##########",
        "#........#",
        "#........#",
        "#...#....#",
        "#...#....#",
        "#........#",
        "#.....#..#",
        "#........#",
        "#........#",
        "##########",
    ]
    grid = []
    for row in layout:
        cells = []
        for symbol in row:
            if symbol == "#":
                cells.append("wall")
            else:
                cells.append("floor")
        grid.append(cells)

    return {
        "grid": grid,
        "objects": [
            {"x": 3, "y": 2, "type": "switch", "state": "off"},
            {"x": 7, "y": 4, "type": "switch", "state": "off"},
            {"x": 2, "y": 7, "type": "switch", "state": "on"},
        ],
        "agent_position": {"x": 2, "y": 2},
        "agent_direction": "east",
    }


def cell_at(world, x, y):
    """Return a fresh description, so saved observations cannot change later."""
    grid = world["grid"]
    if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
        return {"type": "wall"}
    if grid[y][x] == "wall":
        return {"type": "wall"}
    for obj in world["objects"]:
        if obj["x"] == x and obj["y"] == y:
            return {"type": obj["type"], "state": obj["state"]}
    return {"type": "floor"}


def observe(world):
    """Nine named cells, relative to facing. No coordinates leave this function."""
    direction_index = DIRECTIONS.index(world["agent_direction"])
    forward_x, forward_y = FORWARD_OFFSETS[direction_index]
    right_x, right_y = FORWARD_OFFSETS[(direction_index + 1) % 4]
    position = world["agent_position"]

    # Each entry is (name, distance forward, distance right).
    relative_cells = [
        ("ahead_left", 1, -1), ("ahead", 1, 0), ("ahead_right", 1, 1),
        ("left", 0, -1), ("here", 0, 0), ("right", 0, 1),
        ("behind_left", -1, -1), ("behind", -1, 0), ("behind_right", -1, 1),
    ]
    observation = {}
    for name, forward_distance, right_distance in relative_cells:
        x = position["x"] + forward_distance * forward_x + right_distance * right_x
        y = position["y"] + forward_distance * forward_y + right_distance * right_y
        observation[name] = cell_at(world, x, y)
    return observation


def apply_action(world, action):
    """Mutate the physical world and return a readable outcome. No rewards."""
    direction_index = DIRECTIONS.index(world["agent_direction"])
    if action == "turn_left":
        world["agent_direction"] = DIRECTIONS[(direction_index - 1) % 4]
        return "Turned left"
    if action == "turn_right":
        world["agent_direction"] = DIRECTIONS[(direction_index + 1) % 4]
        return "Turned right"

    forward_x, forward_y = FORWARD_OFFSETS[direction_index]
    next_x = world["agent_position"]["x"] + forward_x
    next_y = world["agent_position"]["y"] + forward_y
    ahead = cell_at(world, next_x, next_y)

    if action == "move_forward":
        if ahead["type"] != "floor":
            return "Blocked by " + ahead["type"]
        world["agent_position"] = {"x": next_x, "y": next_y}
        return "Moved forward"

    if action == "interact":
        for obj in world["objects"]:
            if obj["x"] == next_x and obj["y"] == next_y and obj["type"] == "switch":
                if obj["state"] == "off":
                    obj["state"] = "on"
                else:
                    obj["state"] = "off"
                return "Switch changed to " + obj["state"]
        return "Nothing to interact with"

    raise ValueError("Unknown action: " + action)
