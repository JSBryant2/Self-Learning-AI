// Presentation and turn timing only. Python chooses and applies every action.
const startButton = document.getElementById("start");
const stepButton = document.getElementById("single-step");
const resetButton = document.getElementById("reset");
const speedInput = document.getElementById("speed");
const statusLabel = document.getElementById("status");
const errorLabel = document.getElementById("error");
let running = false;
let busy = false; // At most one request is in progress from this browser.
let connected = false;
let timer = null;
let previousDirection = null;
let arrowAngle = 0; // Presentation angle only; never sent to Python.
let logEntries = [];

const observationNames = [
  "ahead_left", "ahead", "ahead_right",
  "left", "here", "right",
  "behind_left", "behind", "behind_right"
];

function readableName(name) {
  return name.replaceAll("_", " ");
}

async function requestState(path, method = "GET") {
  const response = await fetch(path, { method, cache: "no-store" });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Python could not complete the request.");
  }
  return data;
}

function renderWorld(world) {
  const grid = document.getElementById("grid");
  const width = world.grid[0].length;
  const height = world.grid.length;
  grid.style.gridTemplateColumns = `repeat(${width}, 1fr)`;
  grid.style.gridTemplateRows = `repeat(${height}, 1fr)`;
  grid.replaceChildren();

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const cell = document.createElement("div");
      cell.className = "cell " + world.grid[y][x];
      let description = world.grid[y][x];
      for (const object of world.objects) {
        if (object.x === x && object.y === y) {
          const marker = document.createElement("span");
          marker.className = "switch " + object.state;
          cell.append(marker);
          description = object.type + " " + object.state;
        }
      }
      cell.title = description;
      grid.append(cell);
    }
  }

  const marker = document.getElementById("agent");
  marker.style.width = `${100 / width}%`;
  marker.style.height = `${100 / height}%`;
  marker.style.transform = `translate(${world.agent_position.x * 100}%, ${world.agent_position.y * 100}%)`;
  marker.setAttribute("aria-label", "Agent facing " + world.agent_direction);

  // Accumulate small visual rotations so a north/west turn avoids a 270° spin.
  const directions = ["north", "east", "south", "west"];
  const direction = directions.indexOf(world.agent_direction);
  if (previousDirection === null) {
    arrowAngle = direction * 90;
  } else {
    let difference = direction - previousDirection;
    if (difference > 2) difference -= 4;
    if (difference < -2) difference += 4;
    arrowAngle += difference * 90;
  }
  previousDirection = direction;
  document.getElementById("agent-arrow").style.transform = `rotate(${arrowAngle}deg)`;
}

function renderObservation(observation) {
  const container = document.getElementById("observation");
  container.replaceChildren();
  for (const name of observationNames) {
    const description = observation[name];
    const cell = document.createElement("div");
    cell.className = "observation-cell " + description.type;
    if (description.state) cell.classList.add(description.state);
    const label = document.createElement("small");
    label.textContent = readableName(name);
    const content = document.createElement("strong");
    content.textContent = description.type;
    if (description.state) content.textContent += " " + description.state;
    cell.append(label, content);
    container.append(cell);
  }
}

function renderState(state) {
  renderWorld(state.world);
  renderObservation(state.observation);
  document.getElementById("step").textContent = state.step;
  document.getElementById("count").textContent = state.experience_count;
  document.getElementById("action").textContent = state.latest_action ? readableName(state.latest_action) : "No action yet";
  document.getElementById("result").textContent = state.latest_result || "Waiting for the first step";
  renderPrediction(state);
}

function percentage(value) {
  return value === null ? "—" : (value * 100).toFixed(1) + "%";
}

function renderPrediction(state) {
  const experience = state.latest_experience;
  document.getElementById("prediction").textContent = experience ? (experience.prediction ?? "UNKNOWN") : "—";
  document.getElementById("confidence").textContent = experience ? percentage(experience.prediction_confidence) : "—";
  document.getElementById("matching-count").textContent = experience ? experience.prediction_matching_count : 0;

  let correctness = "—";
  if (experience) {
    correctness = experience.prediction_correct === null ? "Not assessed" : (experience.prediction_correct ? "Yes" : "No");
  }
  document.getElementById("prediction-correct").textContent = correctness;
  let context = "No action yet";
  if (experience) {
    const ahead = experience.observation_before.ahead;
    context = ahead.type + (ahead.state ? " " + ahead.state : "") + " ahead · " + readableName(experience.action);
  }
  document.getElementById("prediction-context").textContent = context;

  // These counts come from Python's pre-action search, not the new experience.
  const evidence = document.getElementById("prediction-evidence");
  evidence.replaceChildren();
  if (experience && experience.prediction_matching_count > 0) {
    for (const [outcome, count] of Object.entries(experience.prediction_outcome_counts)) {
      const item = document.createElement("li");
      item.textContent = `${outcome}: ${count} / ${experience.prediction_matching_count}`;
      evidence.append(item);
    }
  } else {
    const item = document.createElement("li");
    item.textContent = experience ? "No matching memories before this action." : "No prediction yet.";
    evidence.append(item);
  }

  const statistics = state.prediction_statistics;
  document.getElementById("accuracy").textContent = percentage(statistics.accuracy);
  document.getElementById("recent-accuracy").textContent = percentage(statistics.recent_accuracy);
  document.getElementById("attempts").textContent = statistics.attempts;
  document.getElementById("predictions-made").textContent = statistics.predictions_made;
  document.getElementById("correct-count").textContent = statistics.correct_predictions;
  document.getElementById("unknown-count").textContent = statistics.unknown_attempts;
  document.getElementById("accuracy-note").textContent = `UNKNOWN attempts are excluded. Recent window: ${statistics.recent_count}/100 known predictions. Statistics survive resets.`;
}

function renderLog() {
  const list = document.getElementById("log");
  list.replaceChildren();
  const entries = logEntries.length ? logEntries : ["No actions yet."];
  for (const entry of entries) {
    const item = document.createElement("li");
    item.textContent = entry;
    list.append(item);
  }
}

function updateControls() {
  startButton.textContent = running ? "Pause" : "Start";
  startButton.disabled = !connected || (busy && !running);
  stepButton.disabled = !connected || running || busy;
  resetButton.disabled = !connected || busy;
  statusLabel.textContent = connected ? (running ? "Running" : "Paused") : "Connection or save error";
}

function pause() {
  running = false;
  clearTimeout(timer);
  updateControls();
}

function showError(error) {
  pause();
  errorLabel.textContent = error.message + " Check that Python is running, then reload this page before continuing.";
  errorLabel.hidden = false;
  connected = false;
  updateControls();
  statusLabel.textContent = "Connection or save error";
}

function scheduleNextStep() {
  // Called only after the previous turn finishes, so requests never overlap.
  timer = setTimeout(performStep, 1000 / Number(speedInput.value));
}

async function performStep() {
  if (busy || !connected) return;
  busy = true;
  updateControls();
  try {
    const state = await requestState("/step", "POST");
    renderState(state);
    const experience = state.latest_experience;
    const correctness = experience.prediction_correct === null ? "not assessed" : (experience.prediction_correct ? "correct" : "incorrect");
    logEntries.unshift(`${state.step} · ${readableName(state.latest_action)} · predicted: ${experience.prediction ?? "UNKNOWN"} (${percentage(experience.prediction_confidence)}) · actual: ${state.latest_result} · ${correctness}`);
    logEntries = logEntries.slice(0, 6);
    renderLog();
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    updateControls();
    if (running) scheduleNextStep();
  }
}

async function resetWorld() {
  if (busy || !connected) return;
  pause();
  busy = true;
  updateControls();
  try {
    const state = await requestState("/reset", "POST");
    renderState(state);
    logEntries = [];
    renderLog();
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    updateControls();
  }
}

startButton.addEventListener("click", () => {
  if (running) {
    pause();
  } else {
    running = true;
    updateControls();
    performStep();
  }
});
stepButton.addEventListener("click", performStep);
resetButton.addEventListener("click", resetWorld);
speedInput.addEventListener("input", () => {
  document.getElementById("speed-label").textContent = speedInput.value + " steps/s";
  if (running && !busy) {
    clearTimeout(timer);
    scheduleNextStep();
  }
});

async function initialise() {
  try {
    const state = await requestState("/state");
    renderState(state);
    connected = true;
    updateControls();
  } catch (error) {
    showError(error);
  }
}

initialise();
