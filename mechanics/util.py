import sympy as sp
from typing import Union, TypeVar
from numbers import Number
from itertools import product, count
import string

T = TypeVar('T')

name_type = Union[str, sp.Symbol]
tuple_ish = Union[T, tuple[T, ...], list[T]]
single_or_tuple = Union[T, tuple[T, ...]]
expr_type = Union[str, sp.Basic, tuple[sp.Basic, ...], list[sp.Basic]]

def split_latex(s, delimiters=(' ', ','), brackets={'(': ')', '{': '}', '[': ']'}):
    result = []
    buf = []
    stack = []  # 括弧の種類を管理

    for char in s:
        if char in brackets.keys():  # 開き括弧
            stack.append(brackets[char])
            buf.append(char)
        elif char in brackets.values():  # 閉じ括弧
            if stack and char == stack[-1]:
                stack.pop()
            buf.append(char)
        elif char in delimiters and not stack:  # トップレベルの区切り文字
            if buf:
                result.append(''.join(buf).strip())
                buf.clear()
        else:
            buf.append(char)

    # 最後に残った文字列
    if buf:
        result.append(''.join(buf).strip())

    return result

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

def is_tuple_ish(items: tuple_ish[T]) -> bool:
    return isinstance(items, (tuple, list))

def to_tuple(items: tuple_ish[T]) -> tuple[T, ...]:
    if isinstance(items, tuple):
        return items
    elif isinstance(items, list):
        return tuple(items)
    else:
        return (items,)
    
def to_single_or_tuple(items: tuple_ish[T], return_as_tuple=None) -> single_or_tuple[T]:
    if isinstance(items, tuple):
        result = items
    elif isinstance(items, list):
        result = tuple(items)
    else:
        result = (items,)

    if return_as_tuple is None:
        if len(result) == 1:
            return result[0]
        else:
            return result
    elif return_as_tuple:
        return result
    else:
        if len(result) == 1:
            return result[0]
        else:
            raise ValueError(f'Expected a single expression, got {len(result)}: {result}')
        
def generate_prefixes():
    """a, b, ..., z, aa, ab, ..., zz, aaa, ..."""

    letters = string.ascii_uppercase
    for n in count(1):  # prefix length
        for comb in product(letters, repeat=n):
            yield ''.join(comb)