from typing import Literal
import sympy as sp

class Space(sp.Symbol):
    pass

class Real(sp.Symbol):
    def __new__(cls):
        return super().__new__(cls, r'\mathbb{R}')

class Sphere(sp.Symbol):
    def __new__(cls):
        return super().__new__(cls, 'S')

class Rotation(sp.Symbol):
    def __new__(cls, dim: int):
        if dim not in (2, 3):
            raise ValueError(f'Rotation space only defined for n=2 or n=3, not {dim}')
        return super().__new__(cls, f'SO({dim})')

R = Real()
S = Sphere()
SO = Rotation