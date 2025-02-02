import math

from mcp.types import ModelPreferences

from mcp_bridge.config import config


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points, ignoring None values.
    """
    valid_dimensions = [
        (p1, p2) for p1, p2 in zip(point1, point2) if p1 is not None and p2 is not None
    ]

    if not valid_dimensions:  # No valid dimensions to compare
        return float("inf")

    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in valid_dimensions))


def find_best_model(preferences: ModelPreferences):
    distance = math.inf
    preffered_model = None
    preference_points = (
        preferences.intelligencePriority,
        preferences.speedPriority,
        preferences.costPriority,
    )

    if preference_points == (None, None, None):
        return config.sampling.models[0]

    for model in config.sampling.models:
        model_points = (model.intelligence, model.speed, model.cost)
        model_distance = euclidean_distance(model_points, preference_points)
        if model_distance < distance:
            distance = model_distance
            preffered_model = model

    if preffered_model is None:
        preffered_model = config.sampling.models[0]

    return preffered_model
