import unittest
import numpy as np
import numpy.testing as npt
from slr import rotate, inverse_rotate, forward_slr, inverse_slr

if __name__ == '__main__':
    unittest.main()


class TestAlg(unittest.TestCase):

    def test_rotate(self):
        a = np.array([1])
        b = np.array([0])
        gamma = 26752.218744  # radian / second / Gauss
        dt = 1e-3
        b1 = np.pi / 2 / dt / gamma
        a, b = rotate(a, b, b1, dt, gamma=gamma)
        npt.assert_allclose(a, 1 / 2**0.5)
        npt.assert_allclose(b, 1j / 2**0.5)

    def test_forward_inverse_slr(self):
        gamma = 26752.218744
        dt = 1e-6
        n = 1000
        b1 = np.pi / 2 / (dt * n) / gamma * np.ones(n)
        a, b = forward_slr(b1, dt)
        b1_inverse = inverse_slr(a, b, dt)
        npt.assert_allclose(b1, b1_inverse)

        b1 = (np.random.randn(n) + 1j * np.random.randn(n)) / (dt * n) / gamma
        a, b = forward_slr(b1, dt)
        b1_inverse = inverse_slr(a, b, dt)
        npt.assert_allclose(b1, b1_inverse)
