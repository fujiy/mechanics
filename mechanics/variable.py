from typing import Optional, Callable, cast
import sympy as sp
import sympy.core.function as spf
import sympy.core.relational as spr

class Variable(spf.AppliedUndef):
    _base_space: tuple[sp.Symbol, ...]
    name: str

    def __new__(cls, *args: sp.Expr, 
                base_space: Optional[tuple[sp.Symbol, ...]] = None, **options):
        var = cast(cls, super().__new__(cls, *args))
        if base_space is None: var._base_space = cast(tuple[sp.Symbol, ...], args)
        else:                  var._base_space = base_space
        return var

    def _sympystr(self, printer) -> str:
        if self._base_space == self.args: return f'{self.name}'
        else:                             return f'{self.name}{self.args}'

    def _latex(self, printer, exp=None) -> str:
        if self._base_space == self.args: 
            latex = f'{self.name}'
        else:                             
            latex = f'{self.name}\\left({", ".join([sp.latex(e) for e in self.args])}\\right)'
        if exp: latex = f'{latex}^{exp}'
        return latex
    
    def diff(self, *args, **kwargs):
        diff = super().diff(*args, **kwargs)
        if isinstance(diff, sp.Derivative):
            print('diff', self, diff)
        return diff
    
    @property
    def base_space(self) -> tuple[sp.Symbol, ...]:
        return self._base_space
    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.__class__(*args, base_space=self.base_space, **options)
    
    
class Function(spf.AppliedUndef):
    _expr:       sp.Expr
    _base_space: tuple[sp.Symbol, ...]
    _is_function: bool
    name: str

    def __new__(cls, *args: sp.Expr, 
                base_space: Optional[tuple[sp.Symbol, ...]] = None, 
                expr: sp.Expr, **options):
        f = cast(cls, super().__new__(cls, *args, **options))

        if base_space is None: f._base_space = cast(tuple[sp.Symbol, ...], args)
        else:                  f._base_space = base_space

        if base_space is None or args == base_space:
            f._expr = expr
        else:
            f._expr = cast(sp.Expr, expr.subs({ old: new for old, new 
                                               in zip(base_space, args) if old != new }))
        f._is_function = True
        return f

    def _sympystr(self, printer) -> str:
        if self.base_space == self.args: return f'{self.name}'
        else:                            return f'{self.name}{self.args}'

    def _latex(self, printer, exp=None) -> str:
        if self.base_space == self.args: 
            latex = f'{self.name}'
        else:
            latex = f'{self.name}\\left({", ".join([sp.latex(e) for e in self.args])}\\right)'
        if exp: latex = f'{latex}^{exp}'
        return latex
    
    @property
    def base_space(self) -> tuple[sp.Symbol, ...]:
        return self._base_space
    
    @property
    def expr(self) -> sp.Expr:
        return self._expr
    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.__class__(*args, base_space=self.base_space, expr=self.expr, **options)
    
class Equation(spr.Equality):
    _label: str

    @property
    def label(self) -> str:
        return self._label
    