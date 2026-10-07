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
            # Immediately ahead of the starting agent, with room for a first push.
            {"x": 3, "y": 2, "type": "block"},
            {"x": 2, "y": 3, "type": "switch", "state": "off"},
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
            description = {"type": obj["type"]}
            if "state" in obj:
                description["state"] = obj["state"]
            return description
    return {"type": "floor"}


def observe(world):
    """Five nearby cells plus far ahead. No coordinates leave this function."""
    direction_index = DIRECTIONS.index(world["agent_direction"])
    forward_x, forward_y = FORWARD_OFFSETS[direction_index]
    right_x, right_y = FORWARD_OFFSETS[(direction_index + 1) % 4]
    position = world["agent_position"]

    # Each entry is (name, distance forward, distance right).
    relative_cells = [
        ("ahead_left", 1, -1), ("ahead", 1, 0), ("ahead_right", 1, 1),
        ("left", 0, -1), ("right", 0, 1),
    ]
    observation = {}
    for name, forward_distance, right_distance in relative_cells:
        x = position["x"] + forward_distance * forward_x + right_distance * right_x
        y = position["y"] + forward_distance * forward_y + right_distance * right_y
        observation[name] = cell_at(world, x, y)

    # Only the far-ahead ray is occluded. Blocks and switches allow sensing beyond.
    if observation["ahead"]["type"] == "wall":
        observation["far_ahead"] = {"type": "unseen"}
    else:
        far_x = position["x"] + 2 * forward_x
        far_y = position["y"] + 2 * forward_y
        observation["far_ahead"] = cell_at(world, far_x, far_y)
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
            if obj["x"] != next_x or obj["y"] != next_y:
                continue
            if obj["type"] == "switch":
                if obj["state"] == "off":
                    obj["state"] = "on"
                else:
                    obj["state"] = "off"
                return "Switch changed to " + obj["state"]
            if obj["type"] == "block":
                destination_x = next_x + forward_x
                destination_y = next_y + forward_y
                destination = cell_at(world, destination_x, destination_y)
                if destination["type"] != "floor":
                    return "Block push blocked by " + destination["type"]
                obj["x"] = destination_x
                obj["y"] = destination_y
                # Pushing moves only the block; the agent stays in place.
                return "Block pushed forward"
        return "Nothing to interact with"

    raise ValueError("Unknown action: " + action)
