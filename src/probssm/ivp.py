import numpy as np


def vanderpol_rhs(t, y, alpha):
    y1, y2 = y
    return np.array([y2, alpha * (1.0 - y1 ** 2) * y2 - y1])


def vanderpol_jac_x(t, y, alpha):
    y1, y2 = y
    return np.array(
        [[0.0, 1.0], [-2.0 * alpha * y2 * y1 - 1.0, alpha * (1.0 - y1 ** 2)]]
    )


def vanderpol_jac_alpha(t, y, alpha):
    y1, y2 = y
    return np.array(
        [
            [0.0],
            [(1.0 - y1 ** 2) * y2],
        ]
    )


def log_rhs(t, y, a, b):
    """RHS for logistic model."""
    return a * y * (1.0 - y / b)


def log_jac(t, y, a, b):
    """Jacobian for logistic model."""
    return np.array([a - a / b * 2.0 * y])


def lv_rhs(t, y, alpha, beta, gamma, delta):
    """RHS for Lotka-Volterra."""
    u, v = y
    return np.array([alpha * u - beta * u * v, -gamma * v + delta * u * v])


def lv_jac_x(t, y, alpha, beta, gamma, delta):
    """Jacobian for Lotka-Volterra."""
    u, v = y
    return np.array([[alpha - beta * v, -beta * u], [delta * v, -gamma + delta * u]])


def lv_jac_alpha(t, y, alpha, beta, gamma, delta):
    u, v = y
    d_dalpha = np.array([[u], [0.0]])
    return d_dalpha


def lv_jac_beta(t, y, alpha, beta, gamma, delta):
    u, v = y
    d_dbeta = np.array([[-u * v], [0.0]])
    return d_dbeta


def lv_jac_gamma(t, y, alpha, beta, gamma, delta):
    u, v = y
    d_dgamma = np.array([[0.0], [-v]])
    return d_dgamma


def lv_jac_delta(t, y, alpha, beta, gamma, delta):
    u, v = y
    d_ddelta = np.array([[0.0], [u * v]])
    return d_ddelta


def sird_rhs(t, y, beta, gamma, eta, population_count):
    """RHS for SIRD model"""

    S, I, R, D = y
    S_next = -beta * S * I / population_count
    I_next = beta * S * I / population_count - gamma * I - eta * I
    R_next = gamma * I
    D_next = eta * I

    return np.array([S_next, I_next, R_next, D_next])


def sird_jac_x(t, y, beta, gamma, eta, population_count):
    """Jacobian for SIRD model w.r.t. the state"""
    S, I, R, D = y
    d_dS = np.array(
        [-beta * I / population_count, -beta * S / population_count, 0.0, 0.0]
    )
    d_dI = np.array(
        [
            beta * I / population_count,
            beta * S / population_count - gamma - eta,
            0.0,
            0.0,
        ]
    )
    d_dR = np.array([0.0, gamma, 0.0, 0.0])
    d_dD = np.array([0.0, eta, 0.0, 0.0])
    jac_matrix = np.array([d_dS, d_dI, d_dR, d_dD])
    return jac_matrix


def sird_jac_beta(t, y, beta, gamma, eta, population_count):
    """Jacobian for SIRD model w.r.t. beta"""
    S, I, R, D = y
    d_dbeta = np.array(
        [[-S * I / population_count], [S * I / population_count], [0.0], [0.0]]
    )
    return d_dbeta


def sirdv_rhs(t, y, V, beta, gamma, eta, population_count):
    """RHS for SIRD model"""

    S, I, R, D = y
    S_next = (-beta * S * I / population_count) - V
    I_next = beta * S * I / population_count - gamma * I - eta * I
    R_next = gamma * I
    D_next = eta * I

    return np.array([S_next, I_next, R_next, D_next])


def sirdv_jac_x(t, y, V, beta, gamma, eta, population_count):
    """Jacobian for SIRD model w.r.t. the state"""
    S, I, R, D = y
    d_dS = np.array(
        [-beta * I / population_count, -beta * S / population_count, 0.0, 0.0]
    )
    d_dI = np.array(
        [
            beta * I / population_count,
            beta * S / population_count - gamma - eta,
            0.0,
            0.0,
        ]
    )
    d_dR = np.array([0.0, gamma, 0.0, 0.0])
    d_dD = np.array([0.0, eta, 0.0, 0.0])

    jac_matrix = np.array([d_dS, d_dI, d_dR, d_dD])
    return jac_matrix


def sirdv_jac_beta(t, y, V, beta, gamma, eta, population_count):
    """Jacobian for SIRD model w.r.t. beta"""
    S, I, R, D = y
    d_dbeta = np.array(
        [
            [-S * I / population_count],
            [S * I / population_count],
            [0.0],
            [0.0],
        ]
    )
    return d_dbeta


def sirdv_jac_vacc(t, y, V, beta, gamma, eta, population_count):
    """Jacobian for SIRD model w.r.t. beta"""
    S, I, R, D = y
    d_dvacc = np.array([[-1.0], [0.0], [0.0], [0.0]])
    return d_dvacc
