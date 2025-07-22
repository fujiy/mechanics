from typing import Literal
import sympy as sp

class Space(sp.Symbol):
    pass

class IntegerSpace(sp.Symbol):
    def __new__(cls, dim: int = 1):
        if dim != 1:
            return super().__new__(cls, fr'\mathbb{{Z}}^{{{dim}}}')
        return super().__new__(cls, r'\mathbb{Z}')

class RealSpace(sp.Symbol):
    def __new__(cls, dim: int = 1):
        if dim != 1:
            return super().__new__(cls, fr'\mathbb{{R}}^{{{dim}}}')
        return super().__new__(cls, r'\mathbb{R}')

class SphereSpace(sp.Symbol):
    def __new__(cls, dim: int = 1):
        if dim != 1:
            return super().__new__(cls, f'S^{{{dim}}}')
        return super().__new__(cls, 'S')

class RotationSpace(sp.Symbol):
    def __new__(cls, dim: int):
        if dim not in (2, 3):
            raise ValueError(f'Rotation space only defined for n=2 or n=3, not {dim}')
        return super().__new__(cls, f'SO({dim})')

Z = IntegerSpace()
R = RealSpace()
S = SphereSpace()
SO = RotationSpace