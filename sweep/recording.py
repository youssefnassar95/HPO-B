"""Keeps the configurations a method proposes during a continuous evaluation."""

import numpy as np


class RecordingMethod:
    """
    Wraps an HPO method and remembers every configuration it suggests.

    HPOBHandler.evaluate_continuous only returns the incumbent curve, so this is
    how a sweep gets hold of the suggested configurations themselves.
    """

    def __init__(self, method):
        self.method = method
        self.suggestions = []

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        if X_pen is not None:
            return self.method.observe_and_suggest(X_obs, y_obs, X_pen)

        x_new = self.method.observe_and_suggest(X_obs, y_obs)
        self.suggestions.append(np.asarray(x_new, dtype=float).reshape(-1).tolist())
        return x_new
