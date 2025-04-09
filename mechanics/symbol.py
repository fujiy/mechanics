from typing import Optional, Callable, cast, Any
import sympy as sp
import sympy.core.function as spf
import sympy.core.relational as spr
from mechanics.util import to_tuple
from mechanics.space import Space

class BaseSpace(sp.Symbol):
    def __new__(cls, name: str, diff_notation = None):
        return super().__new__(cls, name, real=True)
    
    def __init__(self, name: str,diff_notation: None):
        super().__init__()
        self.diff_notation = diff_notation

class Index(sp.Symbol):
    def __new__(cls, name: str, min: Optional[sp.Expr] = None, max: Optional[sp.Expr] = None):
        return super().__new__(cls, name, integer=True)
    
    def __init__(self, name: str, min: Optional[sp.Expr] = None, max: Optional[sp.Expr] = None):
        super().__init__()
        self.min = min
        self.max = max

    def assign(self, value: sp.Expr) -> sp.Expr:
        return value

class Symbol(spf.AppliedUndef):
    name: str
    _base_space: tuple[BaseSpace, ...]
    _space:      Space
    _index:      tuple[Index, ...]
    _iterable:   bool = False

    def _sympystr(self, printer) -> str:
        if self._base_space == self.args: return f'{self.name}'
        else:                             return f'{self.name}{self.args}'

    def _latex(self, printer, exp=None) -> str:
        index_args = self.args[:len(self._index)]
        space_args = self.args[len(self._index):]
        latex = f'{self.name}'
        if index_args: 
            latex += f'_{{{",".join([sp.latex(i) for i in index_args])}}}'
        if space_args != self._base_space:
            latex += f'\\left({",".join([sp.latex(e) for e in space_args])}\\right)'
        if exp: latex = f'{{{latex}}}^{exp}'
        return latex
    

    def __getitem__(self, index: Any) -> 'Symbol':
        raise NotImplementedError('Symbol does not support indexing')
    
    # def __hash__(self):
    #     return hash((self.class_key(), self._base_space, self._index))
    
    @property
    def base_space(self) -> tuple[BaseSpace, ...]:
        return self._base_space
    
    @property
    def space(self) -> Space:
        return self._space
    
    @property
    def index(self) -> tuple[Index, ...]:
        return self._index
    
    @property
    def free_index(self) -> tuple[Index, ...]:
        return tuple(i for n, i in enumerate(self._index) if i != self.args[n])

class Variable(Symbol):

    _is_variable: bool

    @classmethod
    def make(cls, name: str, 
             index: tuple[Index, ...],
             base_space: tuple[BaseSpace, ...],
             space: Space, **options) -> 'Variable':
        args = index + base_space
        return cast(Variable, 
                    spf.UndefinedFunction(name, bases=(Variable,), **options)
                        (*args, index=index, base_space=base_space, space=space, **options)) #type:ignore

    def __new__(cls, *args: sp.Basic, 
                index: tuple[Index, ...] = (),
                base_space: Optional[tuple[BaseSpace, ...]],
                space: Space, **options):
        var = cast(Variable, super().__new__(cls, *args))

        if base_space is None: var._base_space = cast(tuple[BaseSpace, ...], args[len(index):])
        else:                  var._base_space = base_space

        var._index = index
        var._space = space

        var._is_variable = True

        return var
    
    def __getitem__(self, index: Any) -> 'Variable':
        args = list(self.args)
        for n, i in enumerate(to_tuple(index)):
            if n >= len(self._index):
                raise IndexError(f'{str(self)} has only {len(self._index)} indices')
            args[n] = self._index[n].assign(i)
        return self.__class__(*args, index=self._index, base_space=self._base_space, 
                              space=self._space)
    
    # def diff(self, *args, **kwargs):
    #     diff = super().diff(*args, **kwargs)
    #     if isinstance(diff, sp.Derivative):
    #         print('diff', self, diff)
    #     return diff
    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.__class__(*args, index=self._index, base_space=self._base_space,
                               space=self._space, **options)
    
    
class Definition(Symbol):
    _expr: sp.Expr
    _is_definition: bool

    @classmethod
    def make(cls, name: str, 
             index: tuple[Index, ...],
             base_space: tuple[BaseSpace, ...], 
             expr: sp.Expr, condition: sp.Basic = sp.true,
             **options) -> 'Definition':
        args = index + base_space
        return cast(Definition, 
                    spf.UndefinedFunction(name, bases=(Definition,), 
                                          __dict__={'argments': args, 'expr': expr, 'cond': condition})
                                         (*args, index=index, base_space=base_space,  #type: ignore
                                          expr=expr, condition=condition, **options)) #type:ignore

    def __new__(cls, *args: sp.Basic, 
                index: tuple[Index, ...] = (),
                base_space: Optional[tuple[BaseSpace, ...]] = None, 
                expr: sp.Expr, condition: sp.Basic = sp.true, **options):
        f = cast(Definition, super().__new__(cls, *args, **options))

        if base_space is None: f._base_space = cast(tuple[BaseSpace, ...], args)
        else:                  f._base_space = base_space

        f._index = index

        if base_space is None or args[len(base_space):] == base_space:
            expr = expr
        else:
            expr = cast(sp.Expr, expr.subs({ old: new for old, new in zip(base_space, args[len(index):]) if old != new }))
            
        if condition is sp.true:
            f._expr = expr
        else:
            f._expr = sp.Piecewise((expr, condition))

        f._is_definition = True
        return f
    
    def add_definition(self, definition, condition) -> 'Definition':
        
        if condition in [ case[1] for case in self.cases ]:
            raise ValueError(f'Function definition already exists for given condition: {condition}')
        cases = list(self.cases)
        cases.append((definition, condition))

        self._definition = sp.Piecewise(*cases)
        return self

    def __getitem__(self, index: Any) -> 'Definition':
        args = list(self.args)
        for n, i in enumerate(to_tuple(index)):
            if n >= len(self._index):
                raise IndexError(f'{str(self)} has only {len(self._index)} indices')
            args[n] = self._index[n].assign(i)
        return self.__class__(*args, index=self._index, base_space=self._base_space, 
                              expr=cast(sp.Expr, self.expr), condition=sp.true)

    @property
    def expr(self) -> sp.Expr:
        subs = []
        for n, i in enumerate(self.args[:len(self._index)]):
            subs.append((self._index[n], i))
        return self._expr.subs(subs) # type:ignore
    
    @property
    def cases(self) -> tuple[tuple[sp.Expr, sp.Basic], ...]:
        if isinstance(self._expr, sp.Piecewise):
            return cast(tuple[tuple[sp.Expr, sp.Basic], ...], self._expr.args)
        else:
            return ((self._expr, sp.true),)
        
    @property
    def dependency(self) -> tuple[Symbol]:
        return tuple(s for s in self._expr.atoms(spf.AppliedUndef)
                     if isinstance(s, Symbol)) #type:ignore
    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.__class__(*args, index=self._index, base_space=self._base_space,
                               expr=self.expr, condition=sp.true, **options)
    
class Equation(spr.Equality):
    _label: str

    @property
    def label(self) -> str:
        return self._label
    