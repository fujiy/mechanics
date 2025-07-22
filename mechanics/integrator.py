from typing import Optional, Union, cast, Callable, Any
import sympy as sp
import sympy.core.function as spf
from mechanics.util import name_type, expr_type, tuple_ish, to_tuple
from mechanics.symbol import Index, Function, Equation, BaseSpace, Expr
from mechanics.system import System
from mechanics.discretization import DiscretizedSystem, Discretizer

class Integrator(Discretizer):
    name: str
    original: System
    system: DiscretizedSystem
    space: BaseSpace
    index: Index
    X: tuple[Function, ...]
    dX: tuple[Function, ...]
    X_original: tuple[sp.Expr, ...]
    dX_original: tuple[sp.Expr, ...]

    def __init__(self, is_explicit: bool = True):
        super().__init__()

        self.is_explicit = is_explicit
        name = self.__class__.__name__

    def setup(self, original: System, discretized: DiscretizedSystem,
              space: Optional[name_type] = None,
              X: Optional[expr_type] = None, **options):
        
        super().setup(original, discretized, **options)

        if space is None:
            self.space = tuple(discretized._discretized_spaces.keys())[0]
        else:
            self.space = cast(BaseSpace, original(space))

        self.index = discretized._discretized_spaces[self.space]

        if X:
            self.X_original = cast(tuple[sp.Expr, ...], original(X, return_as_tuple=True))
        else:
            self.X_original = original.state_space(self.space)

        self.dX_original = tuple(sp.diff(original(x), self.space) for x in self.X_original)

        self.X = tuple(cast(Function, discretized.discretize_expr(x))
                        for x in self.X_original)
        self.dX = tuple(cast(Function, discretized.discretize_expr(x)) 
                        for x in self.dX_original)
        
    def apply_to_equations(self, equations: tuple[Equation, ...]):

        # nX = set(self.X) - set(self.dX) # non-derivative variables
        # hX = set(self.dX) - set(self.X) # highest order derivatives 

        def replace_equation(X: tuple[sp.Expr, ...], dX: tuple[sp.Expr, ...], label: str, base_index: Optional[Expr] = None):
            if base_index:
                base = {self.index: base_index}
            else:
                base = {}

            # print(f'replace_equation: {X}, {dX}, {label}, {self.dX}')

            for x, x_new in zip(self.X, X):
                # print(f'Updating {x} -> {x_new}')
                if x in self.dX:
                    n = self.dX.index(x)
                    dx_new = dX[n]
                    # print(f'Updating {x} -> {x_new}, {dx_new}')
                    if x_new != dx_new:
                        if hasattr(dx_new, 'name'):
                            name = dx_new.name #type:ignore
                        else:
                            name = str(dx_new)
                        self.system.equate(x_new.subs(base), dx_new.subs(base), label=f'State_{{{name}}}', manipulate=False)

            subs = {}
            for x, x_new, dx, dx_new in zip(self.X, X, self.dX, dX):
                # if dx in hX:
                subs[x] = x_new
                subs[dx] = dx_new

                # if hasattr(dx_new, 'name'):
                #     name = dx_new.name #type:ignore
                # else:
                #     name = str(dx_new)
                # self.system.equate(dx.subs(base), dx_new.subs(base), label=f'State_{{{name}}}', manipulate=False)
                # print(f'subs: {x} -> {x_new}, {dx} -> {dx_new}')

            for eq in equations:
                lhs_d = self.system.discretize_expr(eq.lhs)
                rhs_d = self.system.discretize_expr(eq.rhs)
                variables = self.system.dependencies_of(lhs_d) | self.system.dependencies_of(rhs_d)
                if variables & (set(self.dX) - set(self.X)):
                    lhs_applied = lhs_d.subs(subs).subs(base)
                    rhs_applied = rhs_d.subs(subs).subs(base)
                    self.system.equate(lhs_applied, rhs_applied, label=f"{{{eq.label}}}_{{{label}}}", manipulate=False)
                # diff_n = ''
                # lhs = eq.lhs
                # rhs = eq.rhs
                # while True:
                #     lhs_dicretized = self.system.discretize_expr(lhs)
                #     rhs_dicretized = self.system.discretize_expr(rhs)
                #     variables = self.system.dependencies_of(lhs_dicretized) | self.system.dependencies_of(rhs_dicretized)
                #     if not variables:
                #         break
                #     if variables & (set(self.dX) - set(self.X)):
                #         lhs_applied = lhs_dicretized.subs(subs).subs(base)
                #         rhs_applied = rhs_dicretized.subs(subs).subs(base)

                #         self.system.equate(lhs_applied, rhs_applied, label=f"{{{eq.label}}}_{{{label}}}{diff_n}", manipulate=False)
                #         break

                #     lhs = self.original.diff(lhs, self.space)
                #     rhs = self.original.diff(rhs, self.space)
                #     diff_n += "'"

        for eq in equations:
            diff_n = ''
            lhs = eq.lhs
            rhs = eq.rhs
            while True:
                lhs_d = self.system.discretize_expr(lhs)
                rhs_d = self.system.discretize_expr(rhs)
                variables = self.system.dependencies_of(lhs_d) | self.system.dependencies_of(rhs_d)
                if not variables:
                    break

                if self.is_explicit and variables & (set(self.dX) - set(self.X)):
                    self.system.equate(lhs_d, rhs_d, label=f"{{{eq.label}}}{diff_n}", manipulate=False)
                    break
                
                if not self.is_explicit:
                    if variables & (set(self.dX) - set(self.X)):
                        break
                    self.system.equate(lhs_d, rhs_d, label=f"{{{eq.label}}}{diff_n}", manipulate=False)
                    break


                lhs = self.original.diff(lhs, self.space)
                rhs = self.original.diff(rhs, self.space)
                diff_n += "'"

        self.step_equations(replace_equation)

    def step_equations(self, replace_equation: Callable[[tuple[sp.Expr, ...], tuple[sp.Expr, ...], str], None]):
        raise NotImplementedError('Integrator.step_equations() is not implemented')


    def equation(self, index: Index, X: tuple[Function, ...], F: tuple[sp.Expr, ...])\
        -> list[tuple[sp.Expr, sp.Expr]]:
        raise NotImplementedError('Integrator.equation() is not implemented')



class Euler(Integrator):

    step: expr_type

    def __init__(self, step):
        super().__init__()
        self.step = step
        self.name = 'Euler'

    def step_equations(self, replace_equation: Callable[[tuple[sp.Expr, ...], tuple[sp.Expr, ...], str], None]):
        step = cast(sp.Expr, self.system(self.step))

        K: list[sp.Expr] = []
        for dx in self.dX:
            k_name = f'{{k_{{{dx.name}}}}}'
            self.system.add_variable(k_name, index=tuple(dx.index.keys()))
            K.append(self.system[k_name])

        replace_equation(self.X, tuple(K), 'K')

        self.index.extend_ghost(1)  # Extend index to include the next step

        for x, k in zip(self.X, K):
            self.system.equate(x.at(self.index, self.index + 1), x + step * k, 
                               label=f'{{{self.name}}}_{{{x.name}}}', 
                               manipulate=False)
            
class BackwardEuler(Integrator):

    step: expr_type

    def __init__(self, step):
        super().__init__(is_explicit=False)
        self.step = step
        self.name = 'BEuler'

    def step_equations(self, replace_equation: Callable[[tuple[sp.Expr, ...], tuple[sp.Expr, ...], str], None]):
        step = cast(sp.Expr, self.system(self.step))

        K: list[Function] = []
        for dx in self.dX:
            k_name = f'{{k_{{{dx.name}}}}}'
            self.system.add_variable(k_name, index=tuple(dx.index.keys()))
            K.append(self.system[k_name])

        replace_equation(self.X, tuple(K), 'K')

        self.index.extend_ghost(1)

        for x, k in zip(self.X, K):
            self.system.equate(x.at(self.index, self.index + 1), x + step * k.at(self.index, self.index + 1), 
                               label=f'{{{self.name}}}_{{{x.name}}}', 
                               manipulate=False)
    
class RK2(Integrator):

    step: expr_type

    def __init__(self, step):
        super().__init__()
        self.step = step
        self.name = 'RK2'

    def step_equations(self, replace_equation: Callable[[tuple[sp.Expr, ...], tuple[sp.Expr, ...], str], None]):
        step = cast(sp.Expr, self.system(self.step, manipulate=False))

        K1: list[sp.Expr] = []
        K2: list[sp.Expr] = []
        for dx in self.dX:
            k1_name = f'{{k_{{1{dx.name}}}}}'
            k2_name = f'{{k_{{2{dx.name}}}}}'
            self.system.add_variable(k1_name, index=tuple(dx.index.keys()))
            self.system.add_variable(k2_name, index=tuple(dx.index.keys()))
            K1.append(self.system[k1_name])
            K2.append(self.system[k2_name])

        replace_equation(self.X, tuple(K1), 'K1')
        replace_equation(tuple(x + step * k for x, k in zip(self.X, K1)), tuple(K2), 'K2')

        self.index.extend_ghost(1)

        for x, k1, k2 in zip(self.X, K1, K2):
            self.system.equate(x[self.index + 1], x + step / 2 * (k1 + k2), 
                               label=f'{{{self.name}}}_{{{x.name}}}',
                               manipulate=False)

ImprovedEuler = RK2
Heun = RK2

class RK4(Integrator):

    step: expr_type

    def __init__(self, step):
        super().__init__()
        self.step = step
        self.name = 'RK4'

    def step_equations(self, replace_equation: Callable[[tuple[sp.Expr, ...], tuple[sp.Expr, ...], str], None]):
        step = cast(sp.Expr, self.system(self.step, manipulate=False))

        K1: list[sp.Expr] = []
        K2: list[sp.Expr] = []
        K3: list[sp.Expr] = []
        K4: list[sp.Expr] = []
        for dx in self.dX:
            k1_name = f'{{k_{{1{dx.name}}}}}'
            k2_name = f'{{k_{{2{dx.name}}}}}'
            k3_name = f'{{k_{{3{dx.name}}}}}'
            k4_name = f'{{k_{{4{dx.name}}}}}'
            self.system.add_variable(k1_name, index=tuple(dx.index.keys()))
            self.system.add_variable(k2_name, index=tuple(dx.index.keys()))
            self.system.add_variable(k3_name, index=tuple(dx.index.keys()))
            self.system.add_variable(k4_name, index=tuple(dx.index.keys()))
            K1.append(self.system[k1_name])
            K2.append(self.system[k2_name])
            K3.append(self.system[k3_name])
            K4.append(self.system[k4_name])

        replace_equation(self.X, tuple(K1), 'K1')
        replace_equation(tuple(x + step / 2 * k for x, k in zip(self.X, K1)), tuple(K2), 'K2')
        replace_equation(tuple(x + step / 2 * k for x, k in zip(self.X, K2)), tuple(K3), 'K3')
        replace_equation(tuple(x + step * k for x, k in zip(self.X, K3)), tuple(K4), 'K4')

        self.index.extend_ghost(1)

        for x, k1, k2, k3, k4 in zip(self.X, K1, K2, K3, K4):
            self.system.equate(x[self.index + 1], x + step / 6 * (k1 + 2*k2 + 2*k3 + k4), 
                               label=f'{{{self.name}}}_{{{x.name}}}',
                               manipulate=False)
