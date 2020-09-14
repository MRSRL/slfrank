import sigpy as sp


class Objective(sp.prox.Prox):

    def __init__(self, shape, lamda):
        self.lamda = lamda
        super().__init__(shape)

    def _prox(self, alpha, input):
        n = (len(input) - 1) // 2
        output = input.copy()
        output[1, 0] += alpha
        output[0, 1] += alpha
        output[n + 1, 0] += alpha * self.lamda
        output[0, n + 1] += alpha * self.lamda

        return sp.psd_proj(output)
