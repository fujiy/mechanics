import sympy as sp
from typing import Union, TypeVar
from numbers import Number



T = TypeVar('T')

name_type = Union[str, sp.Symbol]
expr_type = Union[str, sp.Expr, tuple[sp.Expr, ...]]
tuple_ish = Union[T, list[T], tuple[T, ...]]

def make_symbol(name: name_type, **options) -> tuple[sp.Symbol, ...]:
    if isinstance(name, sp.Symbol):
        return (name,)
    elif isinstance(name, (tuple or list)):
        return sum((make_symbol(n, **options) for n in name), tuple())
    elif name == '':
        return tuple()
    else:
        return to_tuple(sp.symbols(name, **options))

def make_function(name: name_type, **options) -> tuple[sp.Function, ...]:
    return to_tuple(sp.symbols(name, **options, cls=sp.Function))

python_name_trans = str.maketrans(
    {'\\': '', '{': '', '}': '', ',': '', ' ': '_', '^': '_', '-': 'm', '\'': 'prime'})
    
def python_name(name: name_type) -> str:
    name = str(name)
    name = name.translate(python_name_trans)
    if name == 'lambda':
        name = 'lambda_'
    return name

def to_tuple(items: tuple_ish[T]) -> tuple[T, ...]:
    if isinstance(items, tuple):
        return items
    elif isinstance(items, list):
        return tuple(items)
    else:
        return (items,)
    