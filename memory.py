"""Individual experiences, not learned knowledge."""

import json


def load_experiences(path):
    experiences = []
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    experiences.append(json.loads(line))
    return experiences


def save_experience(path, experiences, experience):
    # Write first: the in-memory count increases only after saving succeeds.
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(experience) + "\n")
    experiences.append(experience)
