import sigpy as sp
import numpy as np
import numba as nb


class DiagSum(sp.linop.Linop):
    """A Linop that sums along the diagonals of a matrix.
    
    Args:
        n (int): width of matrix.

    """
    def __init__(self, n):
        self.n = n
        super().__init__((2 * n - 1, ), (n, n))

    def _apply(self, input):
        return diag_sum(input)

    def _adjoint_linop(self):
        return DiagEmbed(self.n)


class DiagEmbed(sp.linop.Linop):
    """A Linop that embeds an array along the diagonals of a matrix.
    
    Args:
        n (int): width of matrix.

    """
    def __init__(self, n):
        self.n = n
        super().__init__((n, n), (2 * n - 1, ))

    def _apply(self, input):
        return diag_embed(input)

    def _adjoint_linop(self):
        return DiagSum(self.n)


@nb.jit(cache=True)
def diag_sum(input):
    n = input.shape[0]
    output = np.zeros(2 * n - 1, dtype=input.dtype)
    for i in range(n):
        for j in range(n):
            output[n - 1 + i - j] += input[i, j]

    return output


@nb.jit(cache=True)
def diag_embed(input):
    n = (input.shape[0] + 1) // 2
    output = np.empty((n, n), dtype=input.dtype)
    for i in range(n):
        for j in range(n):
            output[i, j] = input[n - 1 + i - j]

    return output


    
