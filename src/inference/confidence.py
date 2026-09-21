import numpy as np


def classify_with_confidence(probs: np.ndarray, softmax_threshold: float, margin_threshold: float):
    """
    probs: 1D array of class probabilities for a single prediction.
    Returns (predicted_idx, confidence, is_confident: bool).
    An "unknown" verdict is returned when the top prediction is both below
    the absolute softmax threshold AND the margin over the second-best
    guess is too small (the model is genuinely torn between two locations).
    """
    sorted_idx = np.argsort(probs)[::-1]
    top_idx, second_idx = sorted_idx[0], sorted_idx[1]
    top_prob, second_prob = probs[top_idx], probs[second_idx]
    margin = top_prob - second_prob

    is_confident = (top_prob >= softmax_threshold) and (margin >= margin_threshold)
    return int(top_idx), float(top_prob), is_confident