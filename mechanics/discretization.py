from typing import Union, Any, Optional, cast
from collections import defaultdict
import sympy as sp
import sympy.core.function as spf

from .system import System
from .symbol import BaseSpace, Index, Function, Equation
from .util import name_type, expr_type, tuple_ish

class DiscretizedSystem(System):
    _original: System


    _discretized_spaces: dict[BaseSpace, Index]
    _discretized_diffs:  dict[BaseSpace, dict[Function, tuple[Function, int]]]
    _discretizers: list[tuple['Discretizer', dict[str, Any]]]

    def __init__(self, system: System):
        """
        Initialize the Discretization class.

        Args:
            system (System): The system to be discretized.
        """
        super().__init__(space=None)
        self._original = system

        self._discretized_spaces = {}
        self._discretized_diffs = {}
        self._discretizers = []

        for i in self._original._indices:
            self.add_index(i.name, i.min, i.max)

    def uniform_space(self, space: name_type, index: name_type, 
                      min: name_type, max: name_type, step: expr_type) -> 'DiscretizedSystem':
        
        self.add_index(index, min, max)
        index_ = cast(Index, self(index, return_as_tuple=False))
        if space is None: space_ = tuple(self._original._base_space.values())[0]
        else: space_ = cast(BaseSpace, self._original(space, return_as_tuple=False))

        self._discretized_spaces = { space_ : index_ }
        self._discretized_diffs[space_] = {}

        self.define(space_.name, self(step, return_as_tuple=False) * index_) #type: ignore

        return self
    
    def apply(self, discretizer: 'Discretizer', **options) -> 'DiscretizedSystem':
        """
        Apply a discretizer to the system.

        Args:
            discretizer (Discretizer): The discretizer to be applied.

        Returns:
            DiscretizedSystem: The discretized system.
        """
        self._discretizers.append((discretizer, options))
        return self

    def doit(self) -> System:
        """
        Perform the discretization operation.

        Returns:
            System: The discretized system.
        """

        for s in self._original.base_space:
            if s not in self._discretized_spaces:
                self.add_space(s.name)

        for q in self._original.coordinates:
            new_index, new_base_space = self.__discretized_args(q)
            self.add_coordinate(q.name, index=new_index, base_space=new_base_space, space=q.space)
            for space_ in self._discretized_spaces:
                pass
                # self.__replace_diff(q, (space_, 1))
                # self.__replace_diff(q, (space_, 2))

        for v in self._original.variables + self._original.constants:
            new_index, new_base_space = self.__discretized_args(v)
            self.add_variable(v.name, index=new_index, base_space=new_base_space, space=v.space)

        for name, definition in self._original._definitions.items():
            self.define(name, cast(sp.Expr, self.discretize_expr(definition)))

        for discretizer, options in self._discretizers:
            discretizer.setup(self._original, self, **options)

        for discretizer, options in self._discretizers:
            discretizer.apply_to_equations(tuple(self._original._equations.values()))
        if not self._discretizers:
            for eq in self._original._equations.values():
                self.equate(self.discretize_expr(cast(sp.Expr, eq.lhs)), 
                            self.discretize_expr(cast(sp.Expr, eq.rhs)), 
                            label=eq.label)
            
            # if applied: break
            # if not applied:
            #     self.equate(self.discretize_single_expr(cast(sp.Expr, eq.lhs)), 
            #                 self.discretize_expr(cast(sp.Expr, eq.rhs)), 
            #                 label=eq.label)
        
        return self

    def primary_index(self) -> Optional[Index]:
        if len(self._discretized_spaces) == 1:
            return list(self._discretized_spaces.values())[0]
        else:
            return None
        
    def __discretized_args(self, v: Function) -> tuple[tuple[Index, ...], tuple[BaseSpace, ...]]:
        new_index: list[Index] = list(v.index)
        new_base_space: list[BaseSpace] = []
        for s in v.base_space:
            if s in self._discretized_spaces: 
                new_index.append(self._discretized_spaces[s])
            else:           
                new_base_space.append(s)
        return (tuple(new_index), tuple(new_base_space))
    
    def __replace_diff(self, expr: sp.Expr, *args) -> sp.Expr:
        if not isinstance(expr, Function):
            raise ValueError(f'Expression must be a symbol, not {type(expr)}')

        new_index, new_base_space = self.__discretized_args(expr)

        diff_orders = {}
        new_args = []
        for s, n in args:
            if s in self._discretized_spaces:
                diff_orders[s] = n
            else:
                new_args.append((s, n))

        if not diff_orders: 
            return sp.Derivative(expr, *args)


        name = self._original.latex(sp.Derivative(expr, *list(diff_orders.items())))

        if name in self:
            var = self[name]
        else:
            self.add_variable(name, index=new_index, base_space=new_base_space, space=expr.space)
            var = self[name]
            for space_, diff_n in diff_orders.items():
                self._discretized_diffs[space_][cast(Function, var)] = (expr, diff_n)

        if len(new_args) == 0:
            return var
        else:
            return sp.Derivative(var, *new_args)

    
    def discretize_expr(self, expr: sp.Basic) -> sp.Expr:
        def replace_symbol(symbol: Function) -> Function:
            return cast(Function, self[symbol.name])
        return (expr.replace(sp.Derivative, self.__replace_diff) #type:ignore
                    .replace(lambda x: isinstance(x, spf.AppliedUndef), replace_symbol)) #type:ignore

    def state_space(self, time: Optional[name_type] = None) -> tuple[sp.Expr, ...]:
        X = self._original.state_space(time or list(self._discretized_spaces.keys())[0])
        return tuple(cast(Function, self.discretize_expr(x)) for x in X)

class Discretizer:

    def setup(self, original: System, discretized: DiscretizedSystem, **options):
        self.original = original
        self.system = discretized

    def apply_to_equations(self, equations: tuple[Equation, ...]):
        pass 
