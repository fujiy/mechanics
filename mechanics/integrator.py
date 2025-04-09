import sympy as sp
from mechanics.system import Integrator, System
from mechanics.symbol import Index, Symbol
from mechanics.util import expr_type

class Euler(Integrator):

    step: expr_type

    def __init__(self, step):
        self.step = step
        self.name = 'Euler'

        self.is_explicit = True

    # x_{X+1} = X_i + h * F(X_i)
    def equation(self, q: System, index: Index, X: list[Symbol], F: list[sp.Expr])\
        -> list[tuple[sp.Expr, sp.Expr]]:
        step = q(self.step)

        equations = []
        for x, f in zip(X, F):
            equations.append((x[index + 1], x[index] + step * f))

        return equations
    
