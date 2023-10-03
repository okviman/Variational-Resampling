import numpy as np


def lorenz96_discretized_loop(X, F):
    D = len(X)
    X_dot = np.empty(D)
    for i in range(D):
        X_dot[i] = (X[(i + 1) % D] - X[i - 2]) * X[i - 1] - X[i] + F
    return X_dot


def lorenz96_discretized_vectorized(X, F):
    D = len(X)
    ip1 = np.arange(1, D + 1) % D
    im1 = np.arange(-1, D - 1) % D
    im2 = np.arange(-2, D - 2) % D
    X_dot = (X[ip1] - X[im2]) * X[im1] - X + F
    return X_dot


def test_lorenz96_discretized():
    # Test 1: Generic test with random inputs
    D = 10
    F = 8
    X = np.random.rand(D)

    X_dot_loop = lorenz96_discretized_loop(X, F)
    X_dot_vectorized = lorenz96_discretized_vectorized(X, F)

    assert np.allclose(X_dot_loop, X_dot_vectorized), "Test 1 failed"

    # Test 2: Test with zero forcing
    F = 0

    X_dot_loop = lorenz96_discretized_loop(X, F)
    X_dot_vectorized = lorenz96_discretized_vectorized(X, F)

    assert np.allclose(X_dot_loop, X_dot_vectorized), "Test 2 failed"

    # Test 3: Test with specific values (you might have to specify your own values)
    X = np.array([0.5, 0.2, -0.3, -0.7, 1.2])
    F = 7

    X_dot_loop = lorenz96_discretized_loop(X, F)
    X_dot_vectorized = lorenz96_discretized_vectorized(X, F)

    assert np.allclose(X_dot_loop, X_dot_vectorized), "Test 3 failed"

    print("All tests passed!")


# Execute the tests
test_lorenz96_discretized()
