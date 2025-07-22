import datetime
import itertools
import time
from typing import Iterable, cast, Any, Optional
from collections import defaultdict
import tempfile
import os
import importlib
import importlib.resources
import importlib.util
import shortuuid
import sys
import subprocess
import textwrap
from logging import getLogger, DEBUG
import numpy as np
from scipy import linalg
from scipy import sparse
from tqdm import tqdm
import sympy as sp
import sympy.printing.fortran
import matplotlib.pyplot as plt
from sympy.printing.numpy import SciPyPrinter

from mechanics.system import System
from mechanics.symbol import Function, Expr, Index, Equation, Union
from mechanics.util import is_tuple_ish, python_name, name_type, tuple_ish, to_tuple, generate_prefixes
from mechanics.space import Z

logger = getLogger(__name__)

class PythonPrinter(SciPyPrinter):
    # def _print_Symbol(self, expr):
    #         print(expr)
    #         name = super()._print_Symbol(expr)
    #         return name
    
    def _print_AppliedUndef(self, expr):
        if not isinstance(expr, Function):
            raise TypeError(f'Expected Symbol, got {type(expr)}, {expr}')
        if expr.args:
            return python_name(expr.name) + f'[{", ".join(map(str, expr.args))}]'
        else:
            return python_name(expr.name)
        
class FortranPrinter(sympy.printing.fortran.FCodePrinter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.prefixes: dict[str, str] = {}

        self.prefix_gen = generate_prefixes()

    def get_prefix(self, name: str) -> str:
        if name in self.prefixes:
            return self.prefixes[name]
        else:
            prefix = next(self.prefix_gen)
            self.prefixes[name] = prefix
            return prefix

    def _print_Symbol(self, symbol):
        if isinstance(symbol, Index):
            return f'{self.get_prefix(symbol.name)}_{python_name(symbol.name)}'
        else:
            str(symbol)
    
    def _print_Function(self, f):
        if isinstance(f, Function):
            if f.index:
                indices  = [self.doprint(value - i.min + 1) for i, value in f.index.items()]
                return f'{self.get_prefix(f.name)}_{python_name(f.name)}({", ".join(indices)})' 
            else:
                return f'{self.get_prefix(f.name)}_{python_name(f.name)}'
        else:
            return super()._print_Function(f)
        
    def print_name(self, f: Union[Index, Function]) -> str:
        return f'{self.get_prefix(f.name)}_{python_name(f.name)}'
    
    def print_as_array_arg(self, f: Function) -> str:
        if not f.index:
            return f'{self.print_name(f)}'
        return f'{self.print_name(f)}({",".join([":"] * len(f.index))})'
    
    # def print_dimension(self, f: Function) -> str:
    #     if not f.index:
    #         return ''
    #     dimension = 'dimension('
    #     dimension += ', '.join(f'{index.min}:{index.max}' for index in f.index)
    #     dimension += '),'
    #     return dimension
    
        
class Result:
    def __init__(self, system: System, 
                 directory: Optional[str] = None, 
                 name: Optional[str] = None):
        self.system = system

        self.newton_converged_iters = []
        self._dict: dict[str, Any] = {}
        self._builtins = { k: getattr(np, k) for k in dir(np) if not k.startswith('_') }\

        if not directory:
            directory = os.path.join(os.getcwd(), 'result')
        if not name:
            now = datetime.datetime.now()
            name = now.strftime('%Y%m%d_%H%M%S_') + shortuuid.ShortUUID().random(length=6)
           
        self.path = os.path.join(directory, name, '')
            
        os.makedirs(self.path, exist_ok=True)

    def __del__(self):
        for key, value in self._dict.items():
            if isinstance(value, np.memmap):
                value._mmap.close() #type:ignore

    def set_data(self, key: Any, value: Any):
        if isinstance(key, str):
            name = python_name(key)
        elif hasattr(key, 'name'):
            name = python_name(key.name)
        else:
            raise TypeError(f'Expected str or Symbol, got {type(key)}: {key}')
        if name in self._dict:
            raise ValueError(f'Key {name} already exists in the result. ')
        self._dict[name] = value

    def set_data_dict(self, key_values: dict[Any, Any]):
        for key, value in key_values.items():
            self.set_data(key, value)

    # Access

    def __call__(self, expr: str) -> Any:
        data = eval(expr, globals() | self._builtins, self._dict)
        return data
    
    def __getattr__(self, name: str) -> np.ndarray:
        if name in self._dict: return self._dict[name]
        else:                  raise AttributeError(f'\'{name}\' is not exists')
        
    def __getitem__(self, name: str) -> np.ndarray:
        name = python_name(name)
        if name in self._dict: return self._dict[name]
        else:                  raise KeyError(f'\'{name}\' is not exists')

    def __contains__(self, name: str) -> bool:
        name = python_name(name)
        return name in self._dict
    
    def latex(self, expr: str) -> str:
        return self.system.latex(expr)
    
    # For plotting
    def _as_mpl_axes(self) -> Any:
        from mechanics.plot import ResultAxes
        return ResultAxes, { 'result': self } 
        
class Results:
    def __init__(self, results: Iterable[Result] = []):
        self.results: list[Result] = []
        self._builtins = { k: getattr(np, k) for k in dir(np) if not k.startswith('_') }
        self._dict: dict[str, list[Any]] = {}
        for result in results:
            self.append(result)

    def append(self, result: Result):
        for key, value in result._dict.items():
            if key not in self._dict:
                self._dict[key] = [None] * len(self.results)
            self._dict[key].append(value)
        self.results.append(result)

    def __getitem__(self, index: int) -> Result:
        return self.results[index]
    
    def __call__(self, expr: str) -> Any:
        data = eval(expr, globals() | self._builtins, self._dict)
        return data
    
    # For plotting
    def _as_mpl_axes(self) -> Any:
        from mechanics.plot import ResultsAxes
        return ResultsAxes, { 'results': self } 
    
    def latex(self, expr: str) -> str:
        for result in self.results:
            try:
                return result.latex(expr)
            except KeyError:
                continue
        raise KeyError(f'\'{expr}\' is not exists in any results')

class Dependency:

    def __init__(self, var: Function, eq: Equation):
        self.var = var
        self.eq = eq
        self.matching = False

    def __repr__(self) -> str:
        if self.matching:
            return f'{self.var} <-> {self.eq.label}'
        else:
            return f'{self.var} --> {self.eq.label}'
    
class Block:
    def __init__(self, system: System, 
                 id: int, 
                 equations: list[Equation],
                 variables: list[Function],
                 knowns: list[Function],
                 indices: list[Index],
                 depends_on: list['Block']
                 ):
        self.id = id
        self.equations = equations
        self.variables = variables
        self.knowns = knowns
        self.indices = indices
        self.depends_on = depends_on

        self.dim = len(self.equations)
        if self.dim != len(self.variables):
            self.undetermined = True
            logger.debug(f'Block {self.id} has {self.dim} equations, but {len(self.variables)} unknowns')
        else:
            self.undetermined = False
        # assert self.dim == len(self.variables), \
        #     f'Block {self.id} has {self.dim} equations, but {len(self.variables)} unknowns'

        self.explicit = None
        is_linear = True 
        
        for eq in self.equations:
            if not system.is_linear(eq, self.variables):
                is_linear = False
                break

        # if not self.undetermined and (is_linear or self.dim == 1):
        if not self.undetermined and (self.dim == 1):
            # print(f'solving {self.id} explicitly')
            # print(self)
            # for eq in self.equations:
            #     display(eq)
            sol = sp.solve([eq.lhs - eq.rhs for eq in self.equations],  # type:ignore
                           self.variables, dict=True)
            if len(sol) == 1:
                try:
                    self.explicit = {v: sol[0][v] for v in self.variables}
                except KeyError as e:
                    logger.info(f'Block {self.id} has no explicit solution for {e}')
                    self.explicit = None
            # print('solving explicit:', self.explicit)


        self.jacobian: defaultdict[Equation, dict[Function, Expr]] = defaultdict(dict)
        for eq in self.equations:
            derivatives = {}
            for v in variables:
                derivative = sp.diff(eq.lhs - eq.rhs, v)
                if derivative != 0:
                    derivatives[v] = derivative
            if derivatives:
                self.jacobian[eq] = derivatives

        # print(self.jacobian)

    def generate_code(self, printer: FortranPrinter, indices: tuple[Index, ...]) -> str:
        p = printer.doprint
        pn = printer.print_name

        if self.explicit is not None:
            code = f'''

            ! ======================== Block {self.id} ========================
            '''
            for v, expr in self.explicit.items():
                code += f'''
            {p(v)} = {p(expr)} ! {v.name}'''
        
        else:
            code = f'''
            ! ======================== Block {self.id} ========================
            '''
            for i, v in enumerate(self.variables):
                if v.index:
                    cond = ' .and. '.join(f'{p(i.min)} .lt. {p(value)}' for i, value in v.index.items())
                    previous = [(i, value - 1) for i, value in v.index.items()] # type:ignore
                    code += f'''
            if ({cond}) then
                {p(v)} = {p(v.general_form().subs(previous))}
            end if'''

            code += f'''


            do newton_iter = 1, 100
            '''
            for i, eq in enumerate(self.equations):
                code += f'''
                eq({i+1}) = {printer.doprint(eq.lhs - eq.rhs)} ! {eq.label}'''

            code += f'''

                residual = dnrm2({self.dim}, eq, 1)

                if (ieee_is_nan(residual)) then
                    write(message, '("Block {self.id}, nan: ", i5, E10.4)') newton_iter, residual
                    status = 1
                    call flush(6)
                    return
                end if
                if (residual < tol) exit

                jac = 0.0d0

                '''
            for i, eq in enumerate(self.equations):
                for j, v in enumerate(self.variables):
                    if v in self.jacobian[eq]:
                        code += f'''
                jac({i+1}, {j+1}) = {printer.doprint(self.jacobian[eq][v])}'''

            code += f'''
                ! print *, "Block {self.id}"
                ! print *, "eq", eq
                ! print *, "jac", jac

                call dgesv({self.dim}, 1, jac, size(jac, 1), ipiv, eq, {self.dim}, info)
                if (info /= 0) then
                    write(message, '("Block {self.id}, dgesv error: ", i5)') info
                    status = 1
                    call flush(6)
                    return
                end if
                '''
            for i, v in enumerate(self.variables):
                code += f'''
                {p(v)} = {p(v)} - eq({i+1})'''
            code += f'''
            end do
            if (residual >= tol) then
                write(message, '("Block {self.id} not converging: ", E10.4 {"".join([f', " {i.name} =", i5' for i in indices])})') residual {''.join([f", {p(i)}" for i in indices])}
                status = 1
                call flush(6)
                return
            end if
            '''

        return code.replace("&\n", "&\n                ")

    def generate_args(self, printer: FortranPrinter) -> str:
        args = ', '.join([printer.print_name(v) for v in self.indices + self.variables + self.knowns])
        return args

    def __repr__(self) -> str:
        return \
            f'Block #{self.id} {self.indices} <- {tuple(f"#{b.id}" for b in self.depends_on)}\n'\
            f'    equations = {tuple(eq.label for eq in self.equations)}\n'\
            f'    variables = {self.variables}\n'\
            f'    knowns    = {self.knowns}\n'
    
class Stage:
    def __init__(self, inputs: list[Function], blocks: list[Block]):
        self.inputs = inputs
        self.blocks = blocks

    def __repr__(self) -> str:
        s = f'Stage(inputs={self.inputs})):\n'
        for block in self.blocks:
            s += f'  {block}\n'
        return s

Node = Union[Equation, Function]
def show_node(node: Node) -> str:
    if hasattr(node, 'label'):
        return f'{node.label}'
    elif hasattr(node, 'name'):
        return f'{node.name}'
    else:
        return str(node)
class Solver:

    variables: tuple[Function, ...]

    def maximum_matching(self, dependencies: list[Dependency]) \
        -> tuple[list[Dependency], dict[Equation, list[Dependency]], dict[Function, list[Dependency]]]:

        equations: defaultdict[Equation, list[Dependency]] = defaultdict(list)
        unknowns: defaultdict[Function, list[Dependency]] = defaultdict(list)

        for dependency in dependencies:
            equations[dependency.eq].append(dependency)
            unknowns[dependency.var].append(dependency)

        # if len(equations) != len(unknowns):
        #     warnings.warn(f'Number of equations ({len(equations)}) does not match number of unknowns ({len(unknowns)})')

        # assert len(equations) == len(unknowns), \
        #     'Equations seem to be underdetermined or overdetermined: ' \
        #     f'Number of equations ({len(equations)}) does not match number of unknowns ({len(unknowns)})'

        def find_increase_path_from(dep: Dependency, 
                                    searched_eqs: set[Equation] = set(), 
                                    searched_vars: set[Function] = set()) -> list[Dependency]:
            # print(f'find_increase_path_from: {dep}, {searched_eqs}, {searched_vars}')
            eq_matching = False
            for unk_dep in unknowns[dep.var]:
                assert unk_dep.var == dep.var
                if unk_dep.eq == dep.eq: continue
                if unk_dep.eq in searched_eqs: continue

                # print(f'  {dep};  unk_dep: {unk_dep}')
                if unk_dep.matching:
                    eq_matching = True
                    unk_matching = False
                    for more_dep in equations[unk_dep.eq]:
                        assert more_dep.eq == unk_dep.eq
                        if more_dep.var == unk_dep.var: continue
                        if more_dep.var in searched_vars: continue

                        # print(f'  {dep}; more_dep: {more_dep}')
                        unk_matching = True

                        path = find_increase_path_from(more_dep, searched_eqs | {dep.eq}, searched_vars | {unk_dep.var})
                        if path:
                            # print(f'  {dep}; path: {path}')
                            return [dep, unk_dep] + path

                    if not unk_matching:
                        return []

            if not eq_matching:
                return [dep]
            return []

        for eq, deps in equations.items():
            for dep in deps:
                if dep.matching:
                    continue
                path = find_increase_path_from(dep)
                if path:
                    for p in path:
                        p.matching = not p.matching
                    break
            # print(eq.label)

        matched: list[Dependency] = []
        matched_eqs: set[Equation] = set()
        matched_unks: set[Function] = set()
        for dep in dependencies:
            if dep.matching:
                matched.append(dep)
                assert dep.eq not in matched_eqs
                assert dep.var not in matched_unks
                matched_eqs.add(dep.eq)
                matched_unks.add(dep.var)

        if logger.isEnabledFor(DEBUG):
            self.plot_dependencies(dependencies)

        # if len(matched) != len(equations):
        #     warnings.warn(f'Not all equations are matched: {set(equations) - matched_eqs} equations')
        # if len(matched) != len(unknowns):
        #     warnings.warn(f'Not all unknowns are matched: {set(unknowns) - matched_unks} unknowns')
        
        # assert len(matched) == len(equations) == len(unknowns), \
        #     f'Equation {set(equations) - matched_eqs} and {set(unknowns) - matched_unks} is not matched'

        return matched, equations, unknowns

    def block_decomposition(self, dependencies: list[Dependency], inputs: set[Function]) -> list[Block]:
        
        edges: dict[Node, list[Node]] = defaultdict(list)
        inverse_edges: dict[Node, list[Node]] = defaultdict(list)

        for dep in dependencies:
            edges[dep.eq].append(dep.var)
            inverse_edges[dep.var].append(dep.eq)
            if dep.matching:
                edges[dep.var].append(dep.eq)
                inverse_edges[dep.eq].append(dep.var)

        visited: dict[Node, int] = {}
        order = 1

        # print('Edges:', [(f'{show_node(k)} -> {", ".join(show_node(n) for n in v)}') for k, v in edges.items()])
        # print('Inverse edges:', inverse_edges)

        def first_dfs(node: Node):
            nonlocal order
            if node in visited:
                if visited[node] == 0:
                    visited[node] = order
                    order += 1
                return
            else:
                visited[node] = 0

            es = edges.get(node, [])
            if not es:
                visited[node] = order
                order += 1
                return
            for other_node in es:
                first_dfs(other_node)

            if visited[node] == 0:
                visited[node] = order
                order += 1
                    
        for node in edges.keys():
            first_dfs(node)

        visited_ordered = list(visited.items())
        visited_ordered.sort(key=lambda x: x[1], reverse=True)

        # print('Visited order:', visited_ordered)

        visited_second: set[Node] = set()

        def second_dfs(node: Node) -> set[Node]:
            if node in visited_second:
                return set()
            
            visited_second.add(node)

            nodes = set([node])
            for other_node in inverse_edges.get(node, []):
                # print(f'second_dfs: {node} -> {other_node}')
                nodes |= second_dfs(other_node)
            # print('return nodes:', nodes)
            return nodes
        

        nodess: list[set[Node]] = []
        for node, i in visited_ordered:
            nodess.append(second_dfs(node))

        blocks: list[Block] = []
        block_id = 0
        for nodes in reversed(nodess):
            if nodes:
                equations: list[Equation] = []
                variables: list[Function] = []
                knowns: set[Function] = inputs.copy()
                for node in nodes:
                    if isinstance(node, Equation):
                        equations.append(cast(Equation, node))
                        knowns |= set(cast(list[Function], edges.get(node, [])))
                    else:
                        # assert node in self.unknowns
                        variables.append(cast(Function, node))
                knowns = knowns - set(variables)
                depends_on = [ block for block in blocks if set(block.variables) & knowns]
                block = Block(self.system,
                              block_id, 
                              equations, 
                              variables,
                              list(knowns), 
                              list(self.indices),
                              depends_on)
                blocks.append(block)
                block_id += 1
            # print('visited_second', visited_second)

        return blocks

    def __init__(self, system: System, input: Optional[tuple_ish[Function]] = None):

        self.system = system

        self.indices = system.indices
        self.constants = system.constants
        self.coordinates = system.coordinates
        self.variables = system.coordinates + system.variables

        logger.info(f'Indices: {self.indices}')

        self.index_margin = 10

        self.tempdir = tempfile.mkdtemp()

        self.input: tuple[Function, ...] = ()
        if input is None:
            if self.system.indices:
                i = self.system.indices[0]
                self.input = tuple(cast(Function, v.subs(i, i.min)) for v in self.system.state_space())
            else:
                self.input = ()
        else:
            self.input = to_tuple(input)

        logger.info(f'Input: {self.input}')

        constants = set(self.constants)

        self.equations: set[Equation] = set()
        # self.unknowns: set[Function] = set()

        for eq in system.equations.values():
            equation = self.system.eval(eq)
            self.equations.update([cast(Equation, equation)]) 
            # self.unknowns.update(unknowns)

        inputs = set()
        for v in self.input:
            index = []
            for i, i_value in v.index.items():
                if i_value == i.min:
                    index.append(i)
                else:
                    index.append(i_value)
            inputs.add(v[*index])

        index_range_all: defaultdict[Index, set[Expr]] = defaultdict(set)
        for eq in self.equations:
            for var in self.system.dependencies_of(eq):
                for i, i_value in var.index.items():
                    index_range_all[i].add(i_value - i)

        # print(index_range_all)

        self.dependencies: list[Dependency] = []
        for eq in self.equations:
            variables = self.system.dependencies_of(eq)
            index_range = defaultdict(set)
            for var in self.system.dependencies_of(eq):
                for i, i_value in var.index.items():
                    index_range[i].add(i_value - i)

            unknowns = self.system.dependencies_of(eq) - constants  #type:ignore

            patterns = { i: index_range_all[i] - index_range[i] | {0} for i in index_range.keys()}

            index_offsets = itertools.product(*[list(offsets) for offsets in patterns.values()])
            for index_offset in index_offsets:
                index_subs = [(i, i + offset) for i, offset in zip(index_range.keys(), index_offset)]
                # print(f'Index subs for {eq.label}: {list(index_subs)}')
                eq_offset = cast(Equation, eq.subs(index_subs))
                if eq != eq_offset:
                    offset_label = str(",".join(str(i) for i in dict(index_subs).values()))
                    eq_offset._label = f'{eq.label} {offset_label}'
                # print(f'Equation: {eq_offset.label}, {eq_offset}')
                # if python_name(eq_offset._label) in ['State_k_doty', 'State_k_dotx']:
                #     continue
                added = False
                for v in unknowns:
                    v_offset = cast(Function, v.subs(index_subs))
                    if v_offset in inputs:
                        continue
                    self.dependencies.append(Dependency(v_offset, eq_offset))
                    added = True
                if added:
                    # self.system.show(eq_offset, label=eq_offset.label)
                    pass
                    # print(self.system.latex(eq_offset))

        self.maximum_matching(self.dependencies)

        blocks = self.block_decomposition(self.dependencies, inputs)
        block_is_depended_on = defaultdict(set)

        unused_blocks: set[Block] = set()
        for block in reversed(blocks):
            used = False
            if block.id in block_is_depended_on:
                used = True
            for v in block.variables:
                if not v.index:
                    used = True
                if any(i == i_value for i, i_value in v.index.items()):
                    used = True
                if v.general_form() in inputs:
                    used = True
                if used:
                    break
            if used:
                for dep_block in block.depends_on:
                    block_is_depended_on[dep_block.id].add(block)
            else:
                logger.debug(f'Block {block.id} is not used, removing it')
                pass
                unused_blocks.add(block) 
                # print(block)

        blocks = [block for block in blocks if block not in unused_blocks and not block.undetermined]

        orphan_blocks: set[Block] = set()
        while True:
            changed = False
            for block in blocks:
                if any(dep_block not in blocks for dep_block in block.depends_on):
                    orphan_blocks.add(block)
                    logger.debug(f'Block {block.id} is orphan, removing it')
                    changed = True
            if changed:
                blocks = [block for block in blocks if block not in orphan_blocks]
            else:   
                break 

        determined_variables: set[Function] = set()
        for block in blocks:
            for v in block.variables:
                determined_variables.add(v.general_form())

        # for block in blocks:
        #     logger.debug(f'Block {block.id} ({len(block.equations)} equations, {len(block.variables)} variables, {len(block.knowns)} knowns):')
        #     logger.debug(str(block))
                
        nondetermined_variables = set(self.variables) - determined_variables
        if nondetermined_variables:
            raise ValueError(f'Not all variables are determined: {nondetermined_variables}')

        stages: list[Stage] = []

        stage = Stage(list(inputs), blocks)
        
        stages.append(stage)
        logger.debug(str(stage))
        # print(stage)
        if logger.isEnabledFor(DEBUG):
            for stage in stages:
                self.plot_stage(stage)

        printer = FortranPrinter({'source_format': 'free', 'strict': False, 'standard': 95, 'precision': 15})
        p = printer.doprint
        pn = printer.print_name
        
        dim_max = max([block.dim for stage in stages for block in stage.blocks ])
        condition_dim = len(self.constants) + len(self.input)
    
        code = ''
        code += f'''
        module constants
            real(8), save :: pi = 3.14159265358979323846
        '''
        for c in self.constants:
            if c.space == Z:
                code += f'''
            integer, save :: {pn(c)} = 0'''
            else: 
                code += f'''
            real(8), save :: {pn(c)} = 0.0d0'''
        code += f'''
        end module constants
                                
        subroutine run_solver(log_path, condition, status, message)
            use, intrinsic :: ieee_arithmetic
            use constants
            implicit none
            double precision dnrm2
            external dnrm2

            character(len=*), intent(in) :: log_path
            real(8), dimension(:), intent(in) :: condition
            integer, intent(out) :: status
            character(len=100), intent(out) :: message

            integer :: log_unit = 20
            integer :: ios
            
            real(8), dimension({dim_max}) :: eq
            real(8), dimension({dim_max},{dim_max}) :: jac
            real(8), dimension({dim_max}) :: vars
            integer, dimension({dim_max}) :: ipiv
            integer :: info
            real(8) :: residual
            integer :: newton_iter
            real(8) :: tol = 1d-8
            integer :: i

            '''
        
        for i in self.indices:
            code += f'''
            integer :: {pn(i)} = 0'''

        for v in self.variables:
            if v.index:
                code += f'''
            real(8), allocatable :: {printer.print_as_array_arg(v)}'''
            else:
                code += f'''
            real(8) :: {pn(v)} = 0.0d0'''
        code += '\n'

        for f in self.system.definitions.keys():
            if f.index:
                code += f'''
            real(8), allocatable :: {printer.print_as_array_arg(f)}'''
            else:
                code += f'''
            real(8) :: {pn(f)} = 0.0d0'''
        code += '\n'

        for n, c in enumerate(self.constants):
            if c.space == Z:
                code += f'''
            {pn(c)} = int(condition({n + 1}))'''
            else: 
                code += f'''
            {pn(c)} = condition({n + 1})'''
        code += '\n'

        for v in self.variables:
            if v.index:
                code += f'''
            allocate({pn(v)}({",".join(f"{p(i.max - i.min + self.index_margin)}" for i in v.index)}))'''
        code += '\n'

        for v in self.variables:
            code += f'''
            {pn(v)} = 0.0d0'''
        code += '\n'

        for f in self.system.definitions.keys():
            if f.index:
                code += f'''
            allocate({pn(f)}({",".join(f"{p(i.max - i.min + 1)}" for i in f.index)}))'''
        code += '\n'

        for i, v in enumerate(self.input):
            code += f'''
            {p(v)} = condition({len(self.constants) + i + 1})'''
                
        code += f'''        

            print *, "Started"
            print *, "Output in ", log_path

            open(unit=log_unit, file=log_path//"log.bin", form='unformatted',&
                 access='stream', status='replace', iostat=ios)
            if (ios /= 0) then
               print *, "Error opening file: ", log_path//"log.bin"
               stop 1
            end if

        '''
        for i in self.indices:
            code += f'''
            do {pn(i)} = {p(i.min)}, {p(i.max)}
                ! print *, "{pn(i)} =", {pn(i)}
            '''

        for block in stages[0].blocks:
            code += block.generate_code(printer, self.indices)

        for i in self.indices:
            code += f'''
            end do ! {pn(i)}
            '''

        for v in self.variables:
            if v.index:
                code += f'''
            write (log_unit) {pn(v)}({", ".join(f"{1}:{p(i.max - i.min + 1)}" for i in v.index)})'''
            else:
                code += f'''
            write (log_unit) {pn(v)}'''
                
        code += f'''
            print *, "Calculating definitions"
        '''

        for f, definition in self.system.definitions.items():
            for i in self.indices:
                code += f'''
            do {pn(i)} = {p(i.min)}, {p(i.max)}'''
                code += f'''
                {p(f)} = {p(definition)}'''
                code += f'''
            end do ! {pn(i)}'''
            code += f'''
            write (log_unit) {pn(f)} ! {f.name}
            '''
            
        code += f'''

            print *, "Completed"
            call flush(6)
            close(log_unit)
        end subroutine run_solver
        '''
        code = textwrap.dedent(code)
        # print('\n'.join( [ f"{n:04} {line}" for n, line in enumerate(code.splitlines())]))
        # print(self.tempdir)

        self.module = self.compile_and_load(code, [])

    # def __del__(self):
    #     shutil.rmtree(self.tempdir)

    def run(self, 
            condition: dict[name_type, float], 
            directory: Optional[str] = None,
            name: Optional[str] = None,
            newton_max_iter: int = 100, 
            newton_tol: float = 1e-8) -> Result:

        condition_ = { self.system(name): value for name, value in condition.items() }
        # print('Condition:', condition_)

        lack_constants = [c for c in self.constants if c not in condition_]
        if lack_constants:
            raise ValueError(f'Value of {lack_constants} must be provided in condition')

        condition_values = np.array(
            [ condition_[c] for c in self.constants ] 
            + [  condition_[v] for v in self.input ])

        result = Result(self.system, directory=directory, name=name)

        print(condition_values)

        status, message = self.module.run_solver(result.path, condition_values)
        if status != 0:
            time.sleep(0.1)
            raise RuntimeError(message.decode('utf-8'))

        for c in self.constants:
            result.set_data(c, condition_[c])
        
        log_data = np.memmap(os.path.join(result.path, 'log.bin'), dtype=np.float64, mode='r')

        ranges = { i: (sp.sympify(i.min).subs(condition_), sp.sympify(i.max).subs(condition_)) for i in self.indices }
        sizes = { i: r[1] - r[0] + 1 for i, r in ranges.items() }

        for i in self.indices:
            result.set_data(i, range(ranges[i][0], ranges[i][1] + 1))

        offset = 0
        for v in self.variables + tuple(self.system.definitions.keys()):
            if v.index:
                size = np.prod([sizes[i] for i in v.index])
                values = log_data[offset:offset + size].reshape(*[sizes[i] for i in v.index])
                offset += size
                result.set_data(v, values)
            else:
                result.set_data(v, log_data[offset])
                offset += 1
        
        return result
    
    def compile_and_load(self, source: str, libs: list[str] = []) -> Any:

        generate_path = os.path.join(self.tempdir, 'generated.f90')

        with open(generate_path, 'w') as f:
            f.write(source)
        print(f'Generating Fortran code in {generate_path}')

        lib_files = [str(importlib.resources.files('mechanics').joinpath(f'fortran/{filename}'))
            for filename in libs]

        shell_path = subprocess.check_output(['bash', '-l', '-c', 'echo $PATH']).decode().strip()
        env = os.environ.copy()
        env["PATH"] += ':' + shell_path
        generated_name = 'generated'
        ret = subprocess.run([
            sys.executable, '-m', 'numpy.f2py', '-m', generated_name, 
            '-c', 'generated.f90'] + lib_files 
            + ['--build-dir', 'build', '--f90flags="-Wno-unused-dummy-argument"']
            ,
            env=env, cwd=self.tempdir,
            stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE 
            # stdout=subprocess.DEVNULL,
            # stderr=sys.stderr
        )
        if ret.returncode != 0:
            print('======================== Compilation failed ========================')
            # print(ret.stdout.decode())
            raise RuntimeError(f'Compilation failed with return code {ret.returncode}')
        
        sofile = next(p for p in os.listdir(self.tempdir) if p.startswith(generated_name) and p.endswith('.so'))
        so_fullpath = os.path.join(self.tempdir, sofile)

        spec = importlib.util.spec_from_file_location(generated_name, so_fullpath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def plot_dependencies(self, dependencies: list[Dependency]):
        import networkx as nx
        G = nx.MultiDiGraph()

        edge_colors = []
        edge_labels = {}
        edge_curves = []
        unknowns = []
        for i, dep in enumerate(dependencies):
            u = '$' + self.system.latex(dep.var) + '$'
            v = '$' + dep.eq.label + '$'
            G.add_edge(u, v, key=i, matching=dep.matching)
            unknowns.append(dep.var)
            edge_labels[(u, v, i)] = ",".join(str(i) for i in dep.var.index.values())
            edge_curves.append(0.2 * (i))  # for curved edges

        for u, v, k, data in G.edges(keys=True, data=True):
            edge_colors.append('red' if data['matching'] else 'black')

        left_nodes = {'$' + self.system.latex(v) + '$' for v in unknowns}

        plt.figure(figsize=(5, len(left_nodes) * 0.8))
        pos = nx.bipartite_layout(G, left_nodes)
        nx.draw(G, pos, edge_color=edge_colors, with_labels=True,
                connectionstyle=[f"arc3,rad={curve}" for curve in edge_curves], 
                node_size=2000, node_color='lightblue', font_size=10, font_color='black')

        # nx.draw_networkx_edge_labels(G, pos, edge_labels, connectionstyle=[f"arc3,rad={curve}" for curve in edge_curves],  font_size=8, label_pos=0.65)
        # nx.draw_networkx_labels(G, pos)
        plt.title('Dependencies Graph')
        plt.show()

    def plot_stage(self, stage: Stage):
        import networkx as nx
        G = nx.DiGraph()
        labels = {}
        for block in stage.blocks:
            labels[block.id] = \
                f'#{block.id}\n'\
                f'${", ".join(eq.label for eq in block.equations)}$\n'\
                f'${", ".join(self.system.latex(v) for v in block.variables)}$'
        orders = {}
        xs = {}
        for block in stage.blocks:
            if block.depends_on:
                order = max(orders[block.id] for block in block.depends_on) + 1
            else:
                order = 0
            orders[block.id] = order
            
            x = 0
            for id, b in orders.items():
                if b == order:
                    x += 1
            xs[block.id] = x

            G.add_node(block.id)

            # if set(block.knowns) & set(stage.inputs):
            #     G.add_edge('Input', block.id)
            for depends_on in block.depends_on:
                G.add_edge(depends_on.id, block.id)
        color_map = []
        size_map = []
        for block_id in G.nodes:
            block = [block for block in stage.blocks if block.id == block_id][0]
            if block.explicit is None:
                color_map.append('lightcoral')
            else:
                color_map.append('lightblue')
            size_map.append(len(block.equations) * 2500)

        from networkx.algorithms.dag import topological_sort
        import random
        # order = list(topological_sort(G))
        pos = {block.id: (xs[block.id] + random.uniform(-1, 1), -orders[block.id]) for block in stage.blocks}
        # pos = nx.spring_layout(G)

        plt.figure(figsize=(6, 6))
        nx.draw(
            G, pos,
            with_labels=True,
            labels=labels,
            node_color=color_map,
            node_size=size_map,
            font_size=10,
            arrows=True
        )

        plt.title("Block DAG")
        plt.axis('off')
        plt.margins(0.2)
        plt.show()