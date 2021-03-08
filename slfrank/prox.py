import sigpy as sp


class Objective(sp.prox.Prox):

    def __init__(self, shape, lamda):
        self.lamda = lamda
        super().__init__(shape)

    def _prox(self, alpha, input):
        xp = sp.get_array_module(input)
        n = (len(input) - 1) // 2
        output = input.copy()
        output[1, 0] += alpha
        output[n + 1, 0] += alpha * self.lamda

        w, v = xp.linalg.eigh(output)
        w[w < 0] = 0
        return (v * w) @ v.conjugate().T
