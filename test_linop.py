import unittest
import numpy as np
import numpy.testing as npt
from linop import DiagSum

if __name__ == '__main__':
    unittest.main()


class TestLinop(unittest.TestCase):

    def test_DiagSum(self):
        X = np.array([[1, 2], [3, 4]])
        A = DiagSum(2)
        npt.assert_allclose(A(X), [3, 5, 2])
