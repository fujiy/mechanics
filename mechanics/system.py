from typing import Union, Any, Optional, cast
import sympy as sp
from sympy.core.cache import cacheit
import sympy.core.function as spf
import sympy.core.containers as spc
from sympy.printing.latex import LatexPrinter
from mechanics.util import name_type, expr_type, make_symbol, python_name, to_tuple, tuple_ish
from mechanics.symbol import BaseSpace, Index, Symbol, Variable, Definition, Equation
from mechanics.space import R

class System:
    _base_space:  dict[str, BaseSpace]
    _indices:     dict[str, Index]
    _coordinates: dict[str, Variable]
    _variables:   dict[str, Variable]
    _definitions: dict[str, Definition]
    _equations:   dict[str, Equation]
    _dict:        dict[str, Any]
    _builtins:    dict[str, Any]

    _discrete_space: dict[Index, BaseSpace]
    _dicrete_diffs:  dict[BaseSpace, dict[Symbol, tuple[Symbol, int]]]

    def __init__(self, space: Optional[name_type] = 't'):

        self._base_space  = {}
        self._indices     = {}
        self._coordinates = {}
        self._velocities  = {}
        self._variables   = {}
        self._definitions = {}
        self._equations   = {}
        self._dict = {}
        self._discrete_space = {}
        self._dicrete_diffs = {}

        self._builtins = { k: getattr(sp, k) for k in dir(sp) if not k.startswith('_') }\
                       | { 'diff': self.diff }
        
        time_diff_notation = lambda s, n: f'\\{"d" * n}ot{{{s}}}'

        if space: self.add_space(space, diff_notation={'t': time_diff_notation})

    # Basic declaration

    def add_space(self, name: name_type, 
                #   min: Optional[expr_type] = None, 
                #   max: Optional[expr_type] = None,
                  diff_notation = None) -> 'System':
        symbols = make_symbol(name)
        # if bool(min) != bool(max): 
        #     raise ValueError('Both min and max must be provided')
        # if len(symbols) > 1 and min is not None: 
        #     raise ValueError('Only one space variable can be provided with min and max')
        
        for symbol in symbols: 
            # if min is None:
            #     min_ = self.__add_variable(f'{symbol.name}_0', base_space=())[0]
            # else:
            #     min_ = self(min, must_be_single=True)
            # if max is None:
            #     max_ = self.__add_variable(f'{symbol.name}_1', base_space=())[0]
            # else:
            #     max_ = self(max, must_be_single=True)

            if diff_notation is not None:
                if isinstance(diff_notation, dict):
                    diff_notation_ = diff_notation.get(symbol.name, None)
                else:
                    diff_notation_ = diff_notation
            else:
                diff_notation_ = None

            base_space = BaseSpace(symbol.name, diff_notation=diff_notation_)
            self._base_space.update({ symbol.name: base_space })
            self.__register(symbol.name, base_space)

        return self
    
    def add_index(self, name: name_type, min: expr_type, max: expr_type) -> 'System':
        symbols = make_symbol(name)
        for symbol in symbols: 
            index = Index(symbol.name, self(min, must_be_single=True), self(max, must_be_single=True))
            self._indices.update({ symbol.name: index })
            self.__register(symbol.name, index)
        return self

    def add_coordinate(self, name: name_type, 
                       index: Union[name_type, tuple_ish[Index], None] = None,
                       base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                       space: expr_type = R, 
                       **options) -> 'System':
        coordinates = self.__add_variable(name, index=index, base_space=base_space, space=space, **options)
        self._coordinates.update({ q.name: q for q in coordinates})
        return self
    
    def add_variable(self, name: name_type, 
                     index: Union[name_type, tuple_ish[Index], None] = None,
                     base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                     space: expr_type = R, **options) -> 'System':
        self.__add_variable(name, index=index, base_space=base_space, space=space, **options)
        return self
    
    def add_constant(self, name: name_type,
                     index: Union[name_type, tuple_ish[Index], None] = None,
                     **options) -> 'System':
        self.__add_variable(name, index=index, base_space=(), **options)
        return self

    def define(self, name: name_type, expr: expr_type, 
               condition: Union[expr_type, tuple[expr_type, expr_type]] = sp.true,
               index: Optional[expr_type] = None,
               **options) -> 'System':
        symbols = make_symbol(name, **options)
        exprs = self(expr, always_tuple=True, **options)
        if isinstance(condition, tuple):
            condition_ = sp.Eq(self(condition[0]), self(condition[1]))
        elif condition == sp.true:
            condition_ = sp.true
        else:
            condition_ = self(condition, **options)
        if len(symbols) != len(exprs): 
            raise ValueError(f'Number of names and exprs must be the same, {({len(symbols)})} vs {({len(exprs)})}')
        if index:
            index_ = self(index, always_tuple=True)
        else:
            index_ = tuple()

        for symbol, expr in zip(symbols, exprs):
            base_space = self.base_space_of(expr)
            index__ = index_ + self.free_index_of(expr)
            if symbol.name in self._dict:
                self._definitions[symbol.name].add_definition(definition=cast(sp.Expr, expr), condition=condition_)
            else:
                func = Definition.make(symbol.name, index=index__, base_space=base_space,
                                       expr=cast(sp.Expr, expr), condition=condition_)
                self.__register(symbol.name, func)
                self._definitions.update({symbol.name: func})

        return self
    
    def equate(self, expr: expr_type, rhs: expr_type = sp.S.Zero, label: str = '') -> 'System':

        expr = cast(tuple[sp.Expr, ...], self(expr, always_tuple=True))
        rhs  = cast(tuple[sp.Expr, ...], self(rhs, always_tuple=True))
        if len(rhs) == 1: rhs = rhs * len(expr)
        if len(expr) != len(rhs): 
            raise ValueError(f'Number of lhs and rhs must be the same, {({len(expr)})} vs {({len(rhs)})}')

        if label: label_ = to_tuple(label)
        else:     label_ = [f'eq{len(self._equations)}']
        if len(label_) != len(expr): 
            if len(label_) == 1: 
                label_ = [label_[0] + f'_{i}' for i in range(len(expr))]
            else:               
                raise ValueError(f'Number of labels and exprs must be the same, {({len(label_)})} vs {({len(expr)})}')
        
        equations = {} 
        for expr, rhs, l in zip(expr, rhs, label_):
            equation = cast(Equation, Equation(expr, rhs))
            equation._label = l
            self.__register(l, equation)
            equations[l] = equation
        self._equations.update(equations)

        return self
    
    def __register(self, name: name_type, expr: Any):
        name = python_name(name)
        if name in self._dict and expr != self._dict[name]: 
            raise ValueError(f'Name \'{name}\' already exists')
        self._dict[name] = expr

    def __add_variable(self, name: name_type,
                        index: Union[name_type, tuple_ish[Index], None] = None,
                        base_space: Union[name_type, tuple_ish[BaseSpace], None] = None, 
                        space: expr_type = R, **options) \
        -> tuple[Variable, ...]:
        options = { 'real': True } | options
        if base_space is None: base_space_ = tuple(self._base_space.values())
        else:                  base_space_ = self(base_space, always_tuple=True)
        index_ = (index and self(index, always_tuple=True)) or ()
        symbols = make_symbol(name, **options)
        variables = tuple(Variable.make(symbol.name, index=index_, base_space=base_space_, space=space) 
                          for symbol in symbols)
        for var in variables: self.__register(var.name, var) #type:ignore
        self._variables.update({ var.name: var for var in variables })
        return variables
    
    # Advanced declaration
        
    def euler_lagrange_equation(self, L: expr_type, time_var='t', label='EL') -> 'System':
        L_ = self(L, must_be_single=True)
        time_var = self(time_var, must_be_single=True)

        equations: list[sp.Expr] = []

        for q in self.coordinates:
            dLdq = self.diff(L_, q)
            equation = dLdq
            for s in self.base_space:
                dLddq = self.diff(L_, self.diff(q, s))
                d_dLddq_ds = self.diff(dLddq, s)
                equation -= d_dLddq_ds #type:ignore

            equations.append(equation)

        self.equate(tuple(equations), label=label)

        return self
    
    # Get info
    
    def base_space_of(self, expr: expr_type) -> tuple[BaseSpace, ...]:
        exprs = self(expr, always_tuple=True, simplify=False)
        base_space = set().union(*[self.__base_space_of(expr) for expr in exprs])
        return tuple( s for s in self._base_space.values() if s in base_space )

    def __base_space_of(self, expr: sp.Basic) -> set[BaseSpace]:
        base_space = []
        for symbol in expr.atoms(sp.Symbol):
            if symbol.name in self._base_space:
                base_space.append(self._base_space[symbol.name])
        for f in expr.atoms(spf.AppliedUndef):
            name = getattr(f, 'name', None)
            if name in self._variables: 
                base_space.extend(self._variables[name].base_space)
            elif name in self._definitions:
                base_space.extend(self._definitions[name].base_space)
        return set(base_space)
    
    def free_index_of(self, expr: expr_type) -> tuple[Index, ...]:
        exprs = self(expr, always_tuple=True, simplify=False)
        free_index = set().union(*[self.__free_index_of(expr) for expr in exprs])
        return tuple( i for i in self._indices.values() if i in free_index )
    
    def __free_index_of(self, expr: sp.Basic) -> set[Index]:
        free_index = []
        for symbol in expr.atoms(sp.Symbol):
            if symbol.name in self._indices:
                free_index.append(self._indices[symbol.name])
        for f in expr.atoms(spf.AppliedUndef):
            name = getattr(f, 'name', None)
            if name in self._variables: 
                free_index.extend(self._variables[name].free_index)
            elif name in self._definitions:
                free_index.extend(self._definitions[name].free_index)
        return set(free_index)
    
    def is_constant(self, expr: expr_type) -> bool:
        exprs = self(expr, always_tuple=True, simplify=False)
        result = True
        for expr in exprs: #type:ignore
            if isinstance(expr, Variable):
                if expr.base_space or\
                    (expr.index and [i for i in expr.index if i.name in self._indices]):
                    result = False
                    break
        return result

    # Access
    
    def __getattr__(self, name: str) -> sp.Expr:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: str) -> sp.Expr:
        name = python_name(name)
        if name in self._dict: return self._dict[name]
        else:                  raise KeyError(f'\'{name}\' is not exists')

    def __contains__(self, name: str) -> bool:
        name = python_name(name)
        return name in self._dict
        
    # @cacheit
    def __call__(self, expr: expr_type, 
                 sum_for: Optional[name_type]=None,
                 simplify=True, evaluate=False, evaluate_top_level=True,
                 always_tuple=False, must_be_single=False, 
                 ) \
        -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        expr_: Union[sp.Basic, tuple[sp.Basic, ...]]
        if isinstance(expr, str): 
            expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:
            expr_ = expr
        if sum_for:
            index = cast(Index, self(sum_for, must_be_single=True))
            if index.name not in self._indices:
                raise ValueError(f'Index \'{index.name}\' is not exists')
            expr_ = sp.summation(expr_, (index, index.min, index.max))
        if evaluate_top_level and getattr(expr_, '_is_definition', False): 
            expr_ = expr_.expr #type:ignore
        if evaluate:              
            expr_ = self.eval(expr_) #type:ignore
        if isinstance(expr_, spc.Tuple): 
            expr_ = tuple(expr_)
        if isinstance(expr_, list):      
            expr_ = tuple(expr_)
        if simplify: 
            if isinstance(expr_, tuple): 
                expr_ = tuple(sp.simplify(e) for e in expr_)
            else:
                expr_ = sp.simplify(expr_)
        if always_tuple and not isinstance(expr_, tuple):
            expr_ = (expr_,)
        if must_be_single and len(to_tuple(expr_)) != 1: 
            raise ValueError(f'Expression must be a single expression: {expr_}')
        return expr_
    
    # Operations

    def eval(self, expr: expr_type) -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        return self.__eval(expr)

    # @cacheit
    def __eval(self, expr: expr_type, recursive=set()) -> Union[sp.Basic, tuple[sp.Basic, ...]]:
        if isinstance(expr, str): expr_ = eval(expr, globals() | self._builtins, self._dict)
        else:                     expr_ = expr
        if isinstance(expr_, tuple): 
            return tuple(self.__eval(e) for e in expr_) #type:ignore
        evaluated = {}
        for f in expr_.atoms(spf.AppliedUndef):
            if f.name in self._definitions and f not in recursive:
                evaluated[f] = self.__eval(f.expr, recursive | set([f]))

        return cast(sp.Basic, expr_.subs(evaluated))
    

    # @cacheit
    def diff(self, expr, *args, **kwargs) -> sp.Expr:
        return sp.diff(self.eval(expr), 
                       *[self(a, evaluate_top_level=False) for a in args],
                       **kwargs)
    

    # Discretization

    def discretize(self, index: name_type, 
                   space: Union[name_type, tuple_ish[BaseSpace], None] = None,
                   step: Optional[expr_type] = None,
                   ) -> 'System':
        index_ = cast(Index, self(index, must_be_single=True))
        if space is None: space_ = tuple(self._base_space.values())[0]
        else: space_ = cast(BaseSpace, self(space, must_be_single=True))
        
        system = System(space=None)
        system._discrete_space = { index_: space_ }
        system._dicrete_diffs[space_] = {}

        def discretized(symbol: Symbol) -> tuple[tuple[Index, ...], tuple[BaseSpace, ...]]:
            new_index: list[Index] = list(symbol.index)
            new_base_space: list[BaseSpace] = []
            for s in symbol.base_space:
                if s == space_: new_index.append(index_)
                else:           new_base_space.append(s)
            return (tuple(new_index), tuple(new_base_space))
        
        def replace_diff(expr: sp.Expr, *args) -> sp.Expr:

            if isinstance(expr, Symbol):

                new_index, new_base_space = discretized(expr)

                diff_n = 0
                new_args = []
                for s, n in args:
                    if s == space_:
                        diff_n = n
                    else:
                        new_args.append((s, n))

                if diff_n == 0: 
                    return sp.Derivative(expr, *args)

                name = self.latex(sp.Derivative(expr, space_, diff_n))

                if name in self:
                    var = system[name]
                else:
                    system.add_variable(name, index=new_index, base_space=new_base_space, space=expr.space)
                    var = system[name]
                    system._dicrete_diffs[space_][cast(Symbol, var)] = (expr, diff_n)

                if len(new_args) == 0:
                    return var
                else:
                    return sp.Derivative(var, *new_args)

            else:
                raise ValueError(f'Expression must be a symbol, not {type(expr)}')
            
        def replace_symbol(symbol: Symbol) -> Symbol:
            return cast(Symbol, system[symbol.name])
        
        def discretize_expr(expr: sp.Expr) -> sp.Expr:
            return (expr.replace(sp.Derivative, replace_diff) #type:ignore
                        .replace(lambda x: isinstance(x, spf.AppliedUndef), replace_symbol)) #type:ignore


        for s in self._base_space.values():
            if s == space_:
                if step:
                    system.define(s.name, self(step, must_be_single=True) * index_, index=index_)
                else:
                    system.add_variable(s.name, index=index_)
            else: 
                system.add_space(s.name, s.min, s.max)


        for i in self._indices.values():
            system.add_index(i.name, i.min, i.max)

        for q in self.coordinates:
            new_index, new_base_space = discretized(q)
            system.add_coordinate(q.name, index=new_index, base_space=new_base_space, space=q.space)
            replace_diff(q, (space_, 1))
            replace_diff(q, (space_, 2))

        for v in self._variables.values():
            new_index, new_base_space = discretized(v)
            system.add_variable(v.name, index=new_index, base_space=new_base_space, space=v.space)

        for d in self._definitions.values():
            system.define(d.name, cast(sp.Expr, discretize_expr(d.expr)))

        for eq in self._equations.values():
            system.equate(discretize_expr(cast(sp.Expr, eq.lhs)), 
                          discretize_expr(cast(sp.Expr, eq.rhs)), 
                          label=eq.label)

        return system

    def apply_integrator(self, integrator: 'Integrator', index: Optional[Index] = None, 
                         X: Optional[list[Variable]] = None, 
                         F: Optional[list[sp.Expr]] = None) -> 'System':
        if index is None:
            if len(self._discrete_space) == 1:
                index = list(self._discrete_space.keys())[0]
            else:
                raise ValueError('Index must be provided')
        space = self._discrete_space[index]

        if X is None:
            qs = []
            dqs = []
            for q in self.coordinates:
                if index in q.index:
                    qs.append(q)
                    for dq, (q_, n) in self._dicrete_diffs[space].items():
                        if q_.name == q.name and n == 1:
                            dqs.append(cast(Variable, dq))
            
            X = qs + dqs

        if F is None:
            dqs = []
            ddqs = []
            for q in self.coordinates:
                if index in q.index:
                    for dq, (q_, n) in self._dicrete_diffs[space].items():
                        if q_.name == q.name and n == 1:
                            dqs.append(cast(Variable, dq))
                        elif q_.name == q.name and n == 2:
                            ddqs.append(cast(Variable, dq))
            F = dqs + ddqs

        lhs, rhs = zip(*integrator.equation(self, index, X, F))
        self.equate(lhs, rhs, integrator.name)

        return self
    

    # Solver
    def solver(self):
        initial_conditions = list(self.constants)
        
        print(initial_conditions)
    

    # Printing

    def latex(self, expr: expr_type, **options) -> str:
        expr_ = self(expr, evaluate_top_level=False, **options)
        return LatexPrinterModified(self).doprint(expr_)

    def show_all(self) -> 'System':
        from IPython.display import display, Math #type:ignore

        if len(self.coordinates) == 1:
            coords = self.latex(self.coordinates[0])
        else:
            coords = '(' + ', '.join([ self.latex(q) for q in self.coordinates ]) + ')'
        display(Math('Q = ' + self.configuration + r' \ni ' + coords))
        display(Math(r'\mathrm{constants}: ' 
                     + ', '.join([ self.latex(c) for c in self.constants ])))
        display(Math(r'\mathrm{variables}: '
                        + ', '.join([ self.latex(v) for v in self.variables ])))
        display(Math(r'\mathrm{definitions}:'))
        for f in self.definitions:
            display(Math(f'{self.latex(f)} = {self.latex(f.expr)}'))
        display(Math(r'\mathrm{equations}:'))
        for eq in self.equations:
            display(Math(f'\mathrm{{{eq.label}}}: {self.latex(eq.lhs)} = {self.latex(eq.rhs)}'))

        return self
    
    # Properties

    @property
    def base_space(self) -> tuple[sp.Symbol, ...]:
        return tuple(self._base_space.values())
    
    @property
    def indices(self) -> tuple[Index, ...]:
        return tuple(self._indices.values())
    
    @property
    def configuration(self) -> str:
        return r' \times '.join(
            [str(q.space) for q in self._coordinates.values()])
    
    @property
    def coordinates(self) -> tuple[Variable, ...]:
        return tuple(self._coordinates.values())
    
    @property
    def variables(self) -> tuple[Variable, ...]:
        return tuple(v for v in self._variables.values()
                     if not self.is_constant(v) and v.name not in self._coordinates)
    
    @property
    def constants(self) -> tuple[Variable, ...]:
        return tuple(v for v in self._variables.values() 
                     if self.is_constant(v) and v.name not in self._coordinates)
    
    @property
    def definitions(self) -> tuple[Definition, ...]:
        return tuple(self._definitions.values())
    
    @property
    def equations(self) -> tuple[Equation, ...]:
        return tuple(self._equations.values())
    
class Integrator:
    is_explicit: bool
    name: str

    def equation(self, q: System, index: Index, X: list[Symbol], F: list[sp.Expr])\
        -> list[tuple[sp.Expr, sp.Expr]]:
        raise NotImplementedError('Integrator.equation() is not implemented')



class LatexPrinterModified(LatexPrinter):

    def __init__(self, system) -> None:
        super().__init__()
        self._system = system

    def _print_Derivative(self, expr: sp.Expr) -> str:

        notations = []
        no_notations = []

        for s, n in expr.args[1:]: #type:ignore
            if isinstance(s, BaseSpace) \
                and s.name in self._system._base_space \
                and s.diff_notation is not None:
                notations.append((s.diff_notation, n))
            else:
                no_notations.append((s, n))

        printed = self.doprint(expr.args[0])
        for notation, n in notations:
            printed = notation(printed, n)

        if no_notations:
            return super()._print_Derivative(sp.Derivative(sp.Symbol(printed), no_notations)) #type:ignore
        else:
            return printed
