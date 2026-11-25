"""Thirty equations from the Feynman Lectures on Physics, as catalogued by
Udrescu and Tegmark (2020) for AI Feynman, with their identifiers.

Each entry: identifier, target name, formula (SymPy syntax), and the range
each variable is drawn from uniformly. Formulas and ranges were checked
against AI Feynman's table, FeynmanEquations.csv (the copy in the PhySO
repository, commit bfbfa88), which lists I.15.10 and I.39.10 as "I.15.1"
and "I.39.1". The data are generated here; nothing is downloaded.

DEVELOPMENT is the set the search was tuned on. HELD_OUT was chosen before
zsr was ever run on it, to span easy to hard, and was not used for tuning.
"""

DEVELOPMENT = [
    ("I.6.2a", "f", "exp(-theta**2/2)/sqrt(2*pi)", {"theta": (1, 3)}),
    ("I.8.14", "d", "sqrt((x2 - x1)**2 + (y2 - y1)**2)", {"x1": (1, 5), "x2": (1, 5), "y1": (1, 5), "y2": (1, 5)}),
    ("I.10.7", "m", "m_0/sqrt(1 - v**2/c**2)", {"m_0": (1, 5), "v": (1, 2), "c": (3, 10)}),
    ("I.12.1", "F", "mu*N_n", {"mu": (1, 5), "N_n": (1, 5)}),
    ("I.12.2", "F", "q1*q2/(4*pi*epsilon*r**2)", {"q1": (1, 5), "q2": (1, 5), "epsilon": (1, 5), "r": (1, 5)}),
    ("I.12.4", "E_f", "q1/(4*pi*epsilon*r**2)", {"q1": (1, 5), "epsilon": (1, 5), "r": (1, 5)}),
    ("I.13.4", "K", "m*(v**2 + u**2 + w**2)/2", {"m": (1, 5), "v": (1, 5), "u": (1, 5), "w": (1, 5)}),
    ("I.14.3", "U", "m*g*z", {"m": (1, 5), "g": (1, 5), "z": (1, 5)}),
    ("I.14.4", "U", "k_spring*x**2/2", {"k_spring": (1, 5), "x": (1, 5)}),
    ("I.15.10", "p", "m_0*v/sqrt(1 - v**2/c**2)", {"m_0": (1, 5), "v": (1, 2), "c": (3, 10)}),
    ("I.18.4", "r", "(m1*r1 + m2*r2)/(m1 + m2)", {"m1": (1, 5), "m2": (1, 5), "r1": (1, 5), "r2": (1, 5)}),
    ("I.24.6", "E", "m*(omega**2 + omega_0**2)*x**2/4", {"m": (1, 3), "omega": (1, 3), "omega_0": (1, 3), "x": (1, 3)}),
    ("I.25.13", "V_e", "q/C", {"q": (1, 5), "C": (1, 5)}),
    ("I.27.6", "f", "1/(1/d1 + n/d2)", {"d1": (1, 5), "n": (1, 5), "d2": (1, 5)}),
    ("I.29.4", "k", "omega/c", {"omega": (1, 10), "c": (1, 10)}),
    ("I.34.8", "omega", "q*v*B/p", {"q": (1, 5), "v": (1, 5), "B": (1, 5), "p": (1, 5)}),
    ("I.37.4", "I_star", "I1 + I2 + 2*sqrt(I1*I2)*cos(delta)", {"I1": (1, 5), "I2": (1, 5), "delta": (1, 5)}),
    ("I.39.10", "E", "3*pr*V/2", {"pr": (1, 5), "V": (1, 5)}),
    ("I.43.31", "D", "mob*kb*T", {"mob": (1, 5), "kb": (1, 5), "T": (1, 5)}),
    ("II.11.3", "x", "q*E_f/(m*(omega_0**2 - omega**2))", {"q": (1, 3), "E_f": (1, 3), "m": (1, 3), "omega_0": (3, 5), "omega": (1, 2)}),
]

HELD_OUT = [
    ("I.12.5", "F", "q2*E_f", {"q2": (1, 5), "E_f": (1, 5)}),
    ("II.38.3", "F", "Y*A*x/d", {"Y": (1, 5), "A": (1, 5), "d": (1, 5), "x": (1, 5)}),
    ("I.39.22", "pr", "n*kb*T/V", {"n": (1, 5), "T": (1, 5), "V": (1, 5), "kb": (1, 5)}),
    ("I.47.23", "c", "sqrt(gamma*pr/rho)", {"gamma": (1, 5), "pr": (1, 5), "rho": (1, 5)}),
    ("I.16.6", "v1", "(u + v)/(1 + u*v/c**2)", {"c": (1, 5), "v": (1, 5), "u": (1, 5)}),
    ("I.34.1", "omega", "omega_0/(1 - v/c)", {"c": (3, 10), "v": (1, 2), "omega_0": (1, 5)}),
    ("III.17.37", "f", "beta*(1 + alpha*cos(theta))", {"beta": (1, 5), "alpha": (1, 5), "theta": (1, 5)}),
    ("I.48.2", "E", "m*c**2/sqrt(1 - v**2/c**2)", {"m": (1, 5), "v": (1, 2), "c": (3, 10)}),
    ("I.50.26", "x", "x1*(cos(omega*t) + alpha*cos(omega*t)**2)", {"x1": (1, 3), "omega": (1, 3), "t": (1, 3), "alpha": (1, 3)}),
    ("I.40.1", "n", "n_0*exp(-m*g*x/(kb*T))", {"n_0": (1, 5), "m": (1, 5), "x": (1, 5), "T": (1, 5), "g": (1, 5), "kb": (1, 5)}),
]

EQUATIONS = DEVELOPMENT + HELD_OUT
SPLIT = {eq[0]: "development" for eq in DEVELOPMENT} | {eq[0]: "held-out" for eq in HELD_OUT}
