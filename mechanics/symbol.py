from typing import Optional, cast, Any, Union
import itertools
import sympy as sp
import sympy.core.function as spf
import sympy.core.relational as spr
from mechanics.util import to_tuple
from mechanics.space import Space

Expr = sp.Expr
Basic = sp.Basic

class BaseSpace(sp.Symbol):
    min: Optional[Expr]
    max: Optional[Expr]

    def __new__(cls, name: str, diff_notation = None):
        return super().__new__(cls, name, real=True)
    
    def __init__(self, name: str, diff_notation: None):
        super().__init__()
        self.diff_notation = diff_notation
        self.min = None
        self.max = None

class Index(sp.Symbol):
    min: Expr
    max: Expr
    ghosts: list[int]

    def __new__(cls, name: str, min: Expr, max: Expr):
        return super().__new__(cls, name, integer=True)
    
    def __init__(self, name: str, min: Expr, max: Expr):
        super().__init__()
        self.min = min
        self.max = max
        self.ghosts = []

    def assign(self, value: Expr) -> Expr:
        return value
    
    def list(self) -> list[Expr]:
        if self.min is None or self.max is None:
            raise ValueError(f'Index {self} has no min or max value')
        return [self.assign(cast(Expr, i)) for i in range(int(self.min), int(self.max) + 1)]

    def extend_ghost(self, ghost: int) -> 'Index':
        if ghost not in self.ghosts:
            self.ghosts.append(ghost)
        return self
        

class Function(spf.AppliedUndef):
    name: str
    _base_space: tuple[BaseSpace, ...]
    _space:      Space
    _index:      tuple[Index, ...]
    _iterable:   bool = False

    # Initialization

    @classmethod
    def make(cls, name: str, 
             index: tuple[Index, ...],
             base_space: tuple[BaseSpace, ...],
             space: Space,
             args: Optional[tuple[Expr, ...]] = None, **options) -> 'Function':
        if args is None: args = index + base_space
        return cast(Function, 
                    spf.UndefinedFunction(name, bases=(Function,), **options)
                        (*args, index=index, base_space=base_space, space=space, **options)) #type:ignore

    def __new__(cls, *args: Basic, 
                index: tuple[Index, ...] = (),
                base_space: Optional[tuple[BaseSpace, ...]],
                space: Space, **options):
        var = cast(Function, super().__new__(cls, *args))

        if base_space is None: var._base_space = cast(tuple[BaseSpace, ...], args[len(index):])
        else:                  var._base_space = base_space

        var._index = index
        var._space = space

        return var

    
    @property
    def func(self): #type:ignore
        return lambda *args, **options: \
                self.make(self.name, index=self._index, base_space=self._base_space,
                          space=self._space, args=args, **options)
    
    # Indexing

    def __getitem__(self, index: Any) -> 'Function':
        args = cast(list[Expr], list(self.args))
        for n, i in enumerate(to_tuple(index)):
            if n >= len(self._index):
                raise IndexError(f'{str(self)} has only {len(self._index)} indices')
            args[n] = self._index[n].assign(i)
        return self.make(self.name, index=self._index, base_space=self._base_space, 
                         space=self._space, args=tuple(args))
    
    def at(self, index: Index, new_index: Any) -> 'Function':
        if index not in self._index:
            raise ValueError(f'Index {index} not in {self}')
        index_ = self.index
        index_[index] = index.assign(new_index)
        return self[*index_.values()]
    
    def subs_index(self, index_mapping: dict[Index, Expr]) -> 'Function':
        return cast(Function, self.subs(index_mapping))

    def enumerate(self) -> list['Function']:
        if not self._index:
            return [self]
        
        functions = []
        for index in itertools.product(*[i.list() for i in self._index]):
            functions.append(self[*index])
        return functions
    

    def general_form(self) -> 'Function':
        return self.make(self.name, index=self._index, base_space=self._base_space,
                         space=self._space)
    
    def index_mapping(self) -> dict[Index, Expr]:
        return { i: self.index[i] for i in self._index if i != self.index[i] }
    
    def index_matches(self, pattern: 'Function', for_all: list[Index] = []) -> Optional[dict[Index, Expr]]:
        assert self.name == pattern.name, f'Function names do not match: {self.name} != {pattern.name}'
        match = {}
        for i, j in zip(self.index.values(), pattern.index.values()):
            if any(i.coeff(k) not in [0, 1] for k in for_all):
                raise ValueError(f'Index {i} must be simple')
            if any(j.coeff(k) not in [0, 1] for k in for_all):
                raise ValueError(f'Index {j} must be simple')
            minimums = [(k, k.min) for k in for_all]
            maximums = [(k, k.max) for k in for_all]
            min_condition = i.subs(minimums) - j.subs(minimums) >= 0
            max_condition = j.subs(maximums) - i.subs(maximums) >= 0
            if min_condition == True and max_condition == True:
                match[i] = j
            elif min_condition == False or max_condition == False:
                return None
            else:
                raise ValueError(f'Cannot determine index match for {i} and {j} with conditions {min_condition} and {max_condition}')
        return {}

    
    # Get Info

    def is_bound(self, index: Optional[Index]) -> bool:
        if index:
            return index != self.index[index]
        else:
            return any(i != self.index[i] for i in self._index)
        
        
    def is_general_form_of(self, other: 'Function') -> bool:
        if self.name != other.name: return False
        for (i, i_value), (j, j_value) in zip(self.index.items(), other.index.items()):
            if i != j: return False
            if i_value == j_value: continue
            if i == i_value: continue
            return False
        for (s, s_value), (t, t_value) in zip(self.base_space.items(), other.base_space.items()):
            if s != t: return False
            if s_value == t_value: continue
            if s == s_value: continue
            return False 
        return True
   
    # Printing

    def _sympystr(self, printer) -> str:
        if self._base_space == self.args: return f'{self.name}'
        else:                             return f'{self.name}{self.args}'

    def _latex(self, printer, exp=None) -> str:
        index_args = self.args[:len(self._index)]
        space_args = self.args[len(self._index):len(self._index) + len(self._base_space)]
        latex = f'{self.name}'
        if index_args: 
            latex += f'_{{{",".join([sp.latex(i) for i in index_args])}}}'
        if space_args != self._base_space:
            latex += f'\\left({",".join([sp.latex(e) for e in space_args])}\\right)'
        if exp: latex = f'{{{latex}}}^{exp}'
        return latex

    # Properties
      
    @property
    def base_space(self) -> dict[BaseSpace, Expr]:
        return dict(zip(self._base_space, 
                        cast(tuple[Expr, ...], 
                             self.args[len(self._index):len(self._index) + len(self._base_space)])))
    
    @property
    def space(self) -> Space:
        return self._space
    
    @property
    def index(self) -> dict[Index, Expr]:
        return dict(zip(self._index, 
                        cast(tuple[Expr, ...], self.args[:len(self._index)])))
    
    @property
    def free_index(self) -> tuple[Index, ...]:
        return tuple(i for n, i in enumerate(self._index) if i != self.args[n])


    
    @property
    def args_subs(self) -> dict[Union[Index, BaseSpace], Basic]:
        index_args = self.args[:len(self._index)]
        space_args = self.args[len(self._index):]
        return { i: a for i, a in zip(self._index, index_args) if i != a } | \
               { s: a for s, a in zip(self._base_space, space_args) if s != a }

# class Variable(Symbol):

#     _is_variable: bool

#     @classmethod
#     def make(cls, name: str, 
#              index: tuple[Index, ...],
#              base_space: tuple[BaseSpace, ...],
#              space: Space, **options) -> 'Variable':
#         args = index + base_space
#         return cast(Variable, 
#                     spf.UndefinedFunction(name, bases=(Variable,), **options)
#                         (*args, index=index, base_space=base_space, space=space, **options)) #type:ignore

#     def __new__(cls, *args: Basic, 
#                 index: tuple[Index, ...] = (),
#                 base_space: Optional[tuple[BaseSpace, ...]],
#                 space: Space, **options):
#         var = cast(Variable, super().__new__(cls, *args))

#         if base_space is None: var._base_space = cast(tuple[BaseSpace, ...], args[len(index):])
#         else:                  var._base_space = base_space

#         var._index = index
#         var._space = space

#         var._is_variable = True

#         return var
    
#     def __getitem__(self, index: Any) -> 'Variable':
#         args = list(self.args)
#         for n, i in enumerate(to_tuple(index)):
#             if n >= len(self._index):
#                 raise IndexError(f'{str(self)} has only {len(self._index)} indices')
#             args[n] = self._index[n].assign(i)
#         return self.__class__(*args, index=self._index, base_space=self._base_space, space=self._space)
    
#     # def diff(self, *args, **kwargs):
#     #     diff = super().diff(*args, **kwargs)
#     #     if isinstance(diff, sp.Derivative):
#     #         print('diff', self, diff)
#     #     return diff
    
#     @property
#     def func(self): #type:ignore
#         return lambda *args, **options: \
#                 self.__class__(*args, index=self._index, base_space=self._base_space,
#                                space=self._space, **options)
    
    
# class Definition(Symbol):
#     _expr: Expr
#     _is_definition: bool

#     @classmethod
#     def make(cls, name: str, 
#              index: tuple[Index, ...],
#              base_space: tuple[BaseSpace, ...], 
#              expr: Expr, condition: Basic = sp.true,
#              args: Optional[tuple[Basic, ...]] = None,
#              **options) -> 'Definition':
#         if args is None: 
#             args = index + base_space

#         return cast(Definition, 
#                     spf.UndefinedFunction(
#                         name, bases=(Definition,), 
#                         __dict__={'dict_argments': args, 'dict_expr': expr, 'dict_cond': condition})
#                         (*args, index=index, base_space=base_space,  #type: ignore
#                         expr=expr, condition=condition, **options)) #type:ignore
    
#     def __new__(cls, *args: Basic, 
#                 index: tuple[Index, ...] = (),
#                 base_space: Optional[tuple[BaseSpace, ...]] = None, 
#                 expr: Expr, condition: Basic = sp.true, **options):
#         f = cast(Definition, super().__new__(cls, *args, **options))

#         # print('new', id(f), args, index, base_space, expr, condition)
#         # if str(args[0]) == '_d':
#             # raise ValueError('Cannot use _d as a name for a function')

#         if base_space is None: f._base_space = cast(tuple[BaseSpace, ...], args)
#         else:                  f._base_space = base_space

#         f._index = index

#         if base_space is None or args[len(base_space):] == base_space:
#             expr = expr
#         else:
#             expr = cast(Expr, expr.subs({ old: new for old, new in zip(base_space, args[len(index):]) if old != new }))
            
#         if condition is sp.true:
#             f._expr = expr
#         else:
#             f._expr = sp.Piecewise((expr, condition))

#         f._is_definition = True
#         return f
    
#     def add_definition(self, definition, condition) -> 'Definition':
        
#         if condition in [ case[1] for case in self.cases ]:
#             raise ValueError(f'Function definition already exists for given condition: {condition}')
#         cases = list(self.cases)
#         cases.append((definition, condition))

#         self._expr = sp.Piecewise(*cases)
#         return self

#     def __getitem__(self, index: Any) -> 'Definition':
#         args = list(self.args).copy()
#         for n, i in enumerate(to_tuple(index)):
#             if n >= len(self._index):
#                 raise IndexError(f'{str(self)} has only {len(self._index)} indices')
#             args[n] = self._index[n].assign(i)
#         return self.make(self.name, index=self._index, base_space=self._base_space,
#                          expr=cast(Expr, self.single_expr), condition=sp.true, args=tuple(args))
#         # return self.__class__(*args, index=self._index, base_space=self._base_space, 
#         #                       expr=cast(Expr, self.single_expr), condition=sp.true)

#     @property
#     def single_expr(self) -> Expr:
#         subs = []
#         for n, i in enumerate(self.args[:len(self._index)]):
#             subs.append((self._index[n], i))
#         return self._expr.subs(subs) # type:ignore
    
#     @property
#     def cases(self) -> tuple[tuple[Expr, Basic], ...]:
#         if isinstance(self._expr, sp.Piecewise):
#             return cast(tuple[tuple[Expr, Basic], ...], self._expr.args)
#         else:
#             return ((self._expr, sp.true),)
        
#     @property
#     def dependency(self) -> set[Symbol]:
#         return { s for s in self._expr.atoms(spf.AppliedUndef)
#                 if isinstance(s, Symbol) } #type:ignore
    
#     @property
#     def func(self): #type:ignore
#         return lambda *args, **options: \
#                 self.make(self.name, index=self._index, base_space=self._base_space,
#                           expr=self.single_expr, condition=sp.true, args=args, **options)
#                 # cast(Definition, 
#                 #     spf.UndefinedFunction(
#                 #         self.name, bases=(Definition,), 
#                 #         __dict__={'dict_argments': args, 'dict_expr': self.single_expr, 'dict_cond': sp.true})
#                 #         (*args, index=index, base_space=base_space,  #type: ignore
#                 #         expr=self.single_expr, condition=sp.true, **options)) #type:ignore

#                 # self.__class__(*args, index=self._index, base_space=self._base_space,
#                 #                expr=self.single_expr, condition=sp.true, **options)

    
class Equation(spr.Equality):
    _label: str

    @property
    def label(self) -> str:
        return self._label
    
    @property
    def dependency(self) -> set[Function]:
        return { s for s in self.lhs.atoms(spf.AppliedUndef) 
                            | self.rhs.atoms(spf.AppliedUndef)
                if isinstance(s, Symbol) } #type:ignore
    
    def __repr__(self):
        return f'{self._label}: {self.lhs} = {self.rhs}'
    
    @property
    def func(self): #type:ignore
        def make(*args, **options):
            eq = self.__class__(*args, **options)
            eq._label = self._label
            return eq
        return make