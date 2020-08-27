import numpy as np
import sigpy as sp


class Objective(sp.prox.Prox):

    def __init__(self, shape, eps):
        self.eps = eps
        super().__init__(shape)

    def _prox(self, alpha, input):
        n = len(input) // 2
        output = input.copy()
        output[n - 1, n - 1] += alpha
        output[2 * n - 1, n - 1] += alpha * self.eps
        output[n - 1, 2 * n - 1] += alpha * self.eps

        return sp.psd_proj(output)


class MultiBandLinearPhase(sp.prox.Prox):

    def __init__(self, shape, bands):
        self.m = shape[0]
        self.bands = bands
        super().__init__(shape)
