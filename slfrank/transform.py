import numpy as np


def hard_pulse(a, b, b1):
    """Apply hard pulse rotation to input magnetization.

    Args:
        theta (complex float): complex B1 value in radian.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    c = np.cos(abs(b1) / 2)
    if abs(b1) == 0:
        s = 0
    else:
        s = 1j * b1 / abs(b1) * np.sin(abs(b1) / 2)

    return c * a - s.conjugate() * b, s * a + c * b


def inverse_hard_pulse(a, b):
    """
    """
    phi = 2 * np.arctan2(np.abs(b[0]), np.abs(a[0]))
    theta = np.angle(-1j * b[0] * a[0].conjugate())
    b1 = phi * np.exp(1j * theta)
    c = np.cos(abs(b1) / 2)
    if abs(b1) == 0:
        s = 0
    else:
        s = 1j * b1 / abs(b1) * np.sin(abs(b1) / 2)
    return c * a + s.conjugate() * b, -s * a + c * b, b1


def precess(a, b):
    """Simulate precess to input magnetization.

    Args:
        p (array): magnetization array.
        omega (array): off-resonance.
        dt (float): free induction decay duration.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    a = np.concatenate((a, [0]))
    b = np.concatenate(([0], b))
    return a, b


def inverse_precess(a, b):
    """Simulate precess to input magnetization.

    Args:
        p (array): magnetization array.
        omega (array): off-resonance.
        dt (float): free induction decay duration.

    Returns:
        array: magnetization array after hard pulse rotation,
            in representation consistent with input.

    """
    return a[:-1], b[1:]


def forward_slr(b1):
    """Shinnar Le Roux forward evolution.

    The function uses the hard pulse approximation. Given an array of
    B1 complex amplitudes, it simulates a sequence of precess
    followed by a hard pulse rotation.

    Args:
        b1 (array): complex B1 array in Gauss.
        dt (float): delta time in seconds.
        gamma (float): gyromagnetic ratio in radian / seconds / Gauss.

    Returns:
        array: polynomial of shape (n, 2)

    Examples:
        Simulating an on-resonant spin under 90 degree pulse.
        The 90 degree pulse is discretized into 1000 time points.
        1 ms.

        >>> n = 1000
        >>> b1 = np.pi / 2 / n * np.ones(n)
        >>> a, b = forward_slr(b1, dt)

    """
    a, b = hard_pulse(np.array([1]), np.array([0]), b1[0])
    for b1_i in b1[1:]:
        a, b = precess(a, b)
        a, b = hard_pulse(a, b, b1_i)

    return a, b


def inverse_slr(a, b):
    """Shinnar Le Roux inverse evolution.

    The function uses the hard pulse approximation. Given an array of
    B1 complex amplitudes, it simulates a sequence of precess
    followed by a hard pulse rotation.

    Args:
        a (array): alpha polynomial of length n.
        b (array): beta polynomial of length n.

    Returns:
        array: polynomial of shape (n)

    """
    n = len(a)
    b1 = []
    for i in range(n):
        a, b, b1_i = inverse_hard_pulse(a, b)
        a, b = inverse_precess(a, b)
        b1 = [b1_i] + b1

    return np.array(b1)
