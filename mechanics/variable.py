from typing import Optional, Callable, cast, Any
import sympy as sp
import sympy.core.function as spf
import sympy.core.relational as spr
from mechanics.symbol import to_tuple

class BaseSpace(sp.Symbol):
    def __new__(cls, name: str, min: sp.Expr, max: sp.Expr):
        return super().__new__(cls, name, real=True)
    
    def __init__(self, name: str, min: sp.Expr, max: sp.Expr):
        super().__init__()
        self.min = min
        self.max = max

class Index(sp.Symbol):
    def __new__(cls, name: str, min: sp.Expr, max: sp.Expr):
        return super().__new__(cls, name, integer=True)
    
    def __init__(self, name: str, min: sp.Expr, max: sp.Expr):
        super().__init__()
        self.min = min
        self.max = max

    def assign(self, value: sp.Expr) -> sp.Expr:
        return value

class Symbol(spf.AppliedUndef):
    name: str
    _base_space: tuple[BaseSpace, ...]
    _index:      tuple[Index, ...]

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
    
    @property
    def base_space(self) -> tuple[BaseSpace, ...]:
        return self._base_space
    
    @property
    def index(self) -> tuple[Index, ...]:
        return self._index
    
    @property
    def free_index(self) -> tuple[Index, ...]:
        return tuple(i for n, i in enumerate(self._index) if i != self.args[n])

class Variable(Symbol):

    @classmethod
    def make(cls, name: str, 
             index: tuple[Index, ...],
             base_space: tuple[BaseSpace, ...], **options) -> 'Variable':
        args = index + base_space
        return cast(Variable, 
                    spf.UndefinedFunction(name, bases=(Variable,), **options)
                                         (*args, index=index, base_space=base_space, **options)) #type:ignore

    def __new__(cls, *args: sp.Basic, 
                index: tuple[Index, ...] = (),
                base_space: Optional[tuple[BaseSpace, ...]], **options):
        var = cast(Variable, super().__new__(cls, *args))

        if base_space is None: var._base_space = cast(tuple[BaseSpace, ...], args[len(index):])
        else:                  var._base_space = base_space

        var._index = index

        return var
    
    def __getitem__(self, index: Any) -> 'Variable':
        args = list(self.args)
        for n, i in enumerate(to_tuple(index)):
            if n >= len(self._index):
                raise IndexError(f'{str(self)} has only {len(self._index)} indices')
            args[n] = self._index[n].assign(i)
        return self.__class__(*args, index=self._index, base_space=self._base_space)
    
    # def diff(self, *args, **kwargs):
    #     diff = super().diff(*args, **kwargs)
    #     if isinstance(diff, sp.Derivative):
    #         print('diff', self, diff)
    #     return diff
    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.__class__(*args, index=self._index, base_space=self._base_space, **options)
    
    
class Function(Symbol):
    _definition: sp.Expr
    _is_function: bool

    @classmethod
    def make(cls, name: str, 
             index: tuple[Index, ...],
             base_space: tuple[BaseSpace, ...], 
             definition: sp.Expr, condition: sp.Basic = sp.true,
             **options) -> 'Function':
        args = index + base_space
        return cast(Function, 
                    spf.UndefinedFunction(name, bases=(Function,), **options)
                                         (*args, index=index, base_space=base_space,  #type: ignore
                                          definition=definition, condition=condition, **options)) #type:ignore

    def __new__(cls, *args: sp.Basic, 
                index: tuple[Index, ...] = (),
                base_space: Optional[tuple[BaseSpace, ...]] = None, 
                definition: sp.Expr, condition: sp.Basic = sp.true, **options):
        f = cast(Function, super().__new__(cls, *args, **options))

        if base_space is None: f._base_space = cast(tuple[BaseSpace, ...], args)
        else:                  f._base_space = base_space

        f._index = index

        if base_space is None or args[len(base_space):] == base_space:
            definition = definition
        else:
            definition = cast(sp.Expr, 
                              definition.subs({ old: new for old, new in zip(base_space, args[len(index):]) if old != new }))
            
        if condition is sp.true:
            f._definition = definition
        else:
            f._definition = sp.Piecewise((definition, condition))

        f._is_function = True
        return f
    
    def add_definition(self, definition, condition) -> 'Function':
        
        if condition in [ case[1] for case in self.definition_cases ]:
            raise ValueError(f'Function definition already exists for given condition: {condition}')
        cases = list(self.definition_cases)
        cases.append((definition, condition))

        self._definition = sp.Piecewise(*cases)
        return self

    def __getitem__(self, index: Any) -> 'Function':
        args = list(self.args)
        for n, i in enumerate(to_tuple(index)):
            if n >= len(self._index):
                raise IndexError(f'{str(self)} has only {len(self._index)} indices')
            args[n] = self._index[n].assign(i)
        return self.__class__(*args, index=self._index, base_space=self._base_space, 
                              definition=cast(sp.Expr, self.definition), condition=sp.true)

    @property
    def definition(self) -> sp.Expr:
        subs = []
        for n, i in enumerate(self.args[:len(self._index)]):
            subs.append((self._index[n], i))
        return self._definition.subs(subs) # type:ignore
    
    @property
    def definition_cases(self) -> tuple[tuple[sp.Expr, sp.Basic], ...]:
        if isinstance(self._definition, sp.Piecewise):
            return cast(tuple[tuple[sp.Expr, sp.Basic], ...], self._definition.args)
        else:
            return ((self._definition, sp.true),)
    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.__class__(*args, index=self._index, base_space=self._base_space,
                               definition=self.definition, condition=sp.true, **options)
    
class Equation(spr.Equality):
    _label: str

    @property
    def label(self) -> str:
        return self._label
    