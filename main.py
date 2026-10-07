"""Run with python3 main.py, then visit http://127.0.0.1:8000."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from agent import create_agent, choose_action
from environment import create_environment, observe, apply_action
from memory import load_experiences, save_experience

PROJECT_DIRECTORY = Path(__file__).resolve().parent
MEMORY_PATH = PROJECT_DIRECTORY / "experiences.jsonl"


def take_step():
    global step, latest_experience

    observation_before = observe(world)
    action = choose_action(agent, observation_before)
    actual_result = apply_action(world, action)
    observation_after = observe(world)
    step += 1

    experience = {
        "run_id": run_id,
        "step": step,
        "observation_before": observation_before,
        "action": action,
        "actual_result": actual_result,
        "observation_after": observation_after,
    }
    save_experience(MEMORY_PATH, experiences, experience)
    latest_experience = experience
    agent["current_observation"] = observation_after

    print(f"Step {step:4} | {action:12} | {actual_result} | memories: {len(experiences)}", flush=True)
    print("  Observation: " + json.dumps(observation_after), flush=True)


def reset_environment():
    global world, step, run_id, latest_experience
    world = create_environment()
    step = 0
    run_id = str(uuid4())
    latest_experience = None
    # Keep the same agent dictionary and the same experience list.
    agent["current_observation"] = observe(world)
    agent["last_action"] = None
    print(f"Environment reset | retained memories: {len(experiences)}", flush=True)


def display_state():
    # The browser gets the full world. choose_action never receives this payload.
    return {
        "world": world,
        "observation": agent["current_observation"],
        "latest_action": agent["last_action"],
        "latest_result": latest_experience["actual_result"] if latest_experience else None,
        "step": step,
        "experience_count": len(experiences),
    }


class RequestHandler(BaseHTTPRequestHandler):
    """A small adapter required by Python's built-in HTTP server."""

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            self.send_json(display_state())
            return

        # Serve only these three files, rather than exposing the project directory.
        static_files = {
            "/": ("index.html", "text/html"),
            "/style.css": ("style.css", "text/css"),
            "/visualisation.js": ("visualisation.js", "text/javascript"),
        }
        if self.path not in static_files:
            self.send_error(404)
            return
        filename, content_type = static_files[self.path]
        body = (PROJECT_DIRECTORY / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/step", "/reset"):
            self.send_error(404)
            return
        try:
            if self.path == "/step":
                take_step()
            else:
                reset_environment()
            self.send_json(display_state())
        except OSError as error:
            print("Could not save experience: " + str(error), flush=True)
            self.send_json({"error": "Could not save experience. Check the Python console."}, 500)

    def log_message(self, format, *args):
        # Simulation logs above are more useful than a line for every HTTP request.
        pass


def main():
    global agent, experiences
    experiences = load_experiences(MEMORY_PATH)
    agent = create_agent()
    reset_environment()
    server = HTTPServer(("127.0.0.1", 8000), RequestHandler)
    print("Open http://127.0.0.1:8000 in your browser. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Saved experiences remain in experiences.jsonl.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
