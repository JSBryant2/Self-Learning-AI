"""Run with python3 main.py, then visit http://127.0.0.1:8000."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from uuid import uuid4

from agent import create_agent, choose_action
from environment import create_environment, observe, apply_action
from memory import load_experiences, save_experience
from prediction import predict_outcome, prediction_statistics

PROJECT_DIRECTORY = Path(__file__).resolve().parent
MEMORY_PATH = PROJECT_DIRECTORY / "experiences.jsonl"


def take_step():
    global step, latest_experience

    observation_before = observe(world)
    action = choose_action(agent, observation_before)
    # Predict from earlier experiences only. The action has already been chosen.
    prediction = predict_outcome(observation_before, action, experiences)
    actual_result = apply_action(world, action)
    observation_after = observe(world)
    prediction_correct = None
    if prediction["outcome"] is not None:
        prediction_correct = prediction["outcome"] == actual_result
    step += 1

    experience = {
        "run_id": run_id,
        "step": step,
        "observation_before": observation_before,
        "action": action,
        "prediction": prediction["outcome"],
        "prediction_confidence": prediction["confidence"],
        "prediction_matching_count": prediction["matching_count"],
        "prediction_outcome_counts": prediction["outcome_counts"],
        "actual_result": actual_result,
        "observation_after": observation_after,
        "prediction_correct": prediction_correct,
    }
    save_experience(MEMORY_PATH, experiences, experience)
    latest_experience = experience
    agent["current_observation"] = observation_after

    statistics = prediction_statistics(experiences)
    accuracy_text = "—"
    if statistics["accuracy"] is not None:
        accuracy_text = f"{statistics['accuracy']:.1%}"
    correct_text = "not assessed (UNKNOWN)"
    if prediction_correct is not None:
        correct_text = "yes" if prediction_correct else "no"
    ahead = observation_before["ahead"]
    ahead_text = ahead["type"]
    if "state" in ahead:
        ahead_text += " " + ahead["state"]
    observation_text = ahead_text + " ahead"
    if action == "interact" and ahead["type"] == "block":
        far_ahead = observation_before["far_ahead"]
        far_text = far_ahead["type"]
        if "state" in far_ahead:
            far_text += " " + far_ahead["state"]
        observation_text += "; " + far_text + " far ahead"
    print(
        f"\nSTEP {step}\n"
        f"Observation before action: {observation_text}\n"
        f"Action: {action}\n"
        f"Prediction: {prediction['outcome'] or 'UNKNOWN'}\n"
        f"Confidence: {prediction['confidence']:.1%}\n"
        f"Matching experiences: {prediction['matching_count']}\n"
        f"Outcome counts: {json.dumps(prediction['outcome_counts'])}\n"
        f"Actual result: {actual_result}\n"
        f"Prediction correct: {correct_text}\n"
        f"Running prediction accuracy: {accuracy_text} "
        f"({statistics['correct_predictions']}/{statistics['predictions_made']} known predictions)\n"
        f"Stored experiences: {len(experiences)}",
        flush=True,
    )


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
        "latest_experience": latest_experience,
        "prediction_statistics": prediction_statistics(experiences),
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
