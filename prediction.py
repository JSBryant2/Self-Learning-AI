"""Infer immediate outcomes from memories. This never chooses an action."""


def predict_outcome(observation, action, experiences):
    # Switch state is part of ahead. Block pushes also depend on the destination.
    ahead = observation["ahead"]
    needs_far_ahead = action == "interact" and ahead["type"] == "block"
    outcome_counts = {}
    matching_count = 0

    for experience in experiences:
        if experience["action"] != action:
            continue
        previous_observation = experience["observation_before"]
        if previous_observation["ahead"] != ahead:
            continue
        if needs_far_ahead:
            # Older memories without this information cannot explain a block push.
            if "far_ahead" not in observation or "far_ahead" not in previous_observation:
                continue
            if previous_observation["far_ahead"] != observation["far_ahead"]:
                continue

        outcome = experience["actual_result"]
        if outcome not in outcome_counts:
            outcome_counts[outcome] = 0
        outcome_counts[outcome] += 1
        matching_count += 1

    predicted_outcome = None  # None means UNKNOWN, not an observed outcome.
    most_common_count = 0
    for outcome, count in outcome_counts.items():
        # Strictly greater keeps the first encountered outcome when counts tie.
        if count > most_common_count:
            predicted_outcome = outcome
            most_common_count = count

    confidence = 0.0
    if matching_count > 0:
        confidence = most_common_count / matching_count

    return {
        "outcome": predicted_outcome,
        "confidence": confidence,
        "matching_count": matching_count,
        "outcome_counts": outcome_counts,
    }


def prediction_statistics(experiences):
    # Recompute from saved records, so reset and restart retain the same metrics.
    attempts = 0
    correct_count = 0
    known_results = []
    for experience in experiences:
        if "prediction" not in experience:
            continue  # Version 1 memories are evidence, but were not predictions.
        attempts += 1
        if experience["prediction"] is None:
            continue  # UNKNOWN is neither correct nor incorrect.
        correct = experience["prediction_correct"]
        known_results.append(correct)
        if correct:
            correct_count += 1

    predictions_made = len(known_results)
    accuracy = None
    if predictions_made > 0:
        accuracy = correct_count / predictions_made

    recent_results = known_results[-100:]
    recent_correct_count = 0
    for correct in recent_results:
        if correct:
            recent_correct_count += 1
    recent_accuracy = None
    if recent_results:
        recent_accuracy = recent_correct_count / len(recent_results)

    return {
        "attempts": attempts,
        "predictions_made": predictions_made,
        "unknown_attempts": attempts - predictions_made,
        "correct_predictions": correct_count,
        "accuracy": accuracy,
        "recent_accuracy": recent_accuracy,
        "recent_count": len(recent_results),
    }
