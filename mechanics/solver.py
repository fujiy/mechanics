import datetime
import itertools
from typing import cast, Any, Optional
from collections import defaultdict
import tempfile
import os
import importlib
import importlib.resources
import importlib.util
import shutil
import sys
import subprocess
import textwrap
import warnings
import numpy as np
from scipy import linalg
from scipy import sparse
from tqdm import tqdm
import sympy as sp
import sympy.printing.fortran
import matplotlib.pyplot as plt
from sympy.printing.numpy import SciPyPrinter
import networkx as nx

from mechanics.system import System
from mechanics.symbol import Function, Expr, Index, Equation, Union
from mechanics.util import python_name, name_type, tuple_ish, to_tuple, generate_prefixes
from mechanics.space import Z

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
        self._dict = {}

        if not directory:
            directory = os.path.join(os.getcwd(), 'result')
        if not name:
            now = datetime.datetime.now()
            name = now.strftime('%Y%m%d_%H%M%S')
           
        self.path = os.path.join(directory, name, '')
            
        os.makedirs(self.path, exist_ok=True)

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
    
    # For plotting
    def _as_mpl_axes(self) -> Any:
        from mechanics.plot import ResultAxes
        return ResultAxes, { 'result': self } 
        

class Dependency:

    def __init__(self, eq: Equation, var: Function, index_mapping: dict[Index, Expr]):
        self.eq = eq
        self.var = var
        self.index_mapping = index_mapping # index of eq <-> index of var
        self.matching = False

    def __repr__(self) -> str:
        return f'{self.eq.label} <- {self.var.name}, {self.index_mapping}'
    
class Block:
    def __init__(self, id: int, 
                 equations: list[Equation],
                 variables: list[Function],
                 unknowns: list[Function], 
                 knowns: list[Function],
                 indices: list[Index]):
        self.id = id
        self.equations = equations
        self.variables = variables
        self.unknowns = unknowns
        self.knowns = knowns
        self.indices = indices

        self.dim = len(self.equations)

        self.jacobian: defaultdict[Equation, dict[Function, Expr]] = defaultdict(dict)
        for eq in self.equations:
            derivatives = {}
            for v in unknowns:
                derivative = sp.diff(eq.lhs - eq.rhs, v)
                if derivative != 0:
                    derivatives[v] = derivative
            if derivatives:
                self.jacobian[eq] = derivatives

        print(self.jacobian)

    def generate_fortran_args(self, printer: FortranPrinter) -> str:
        args = ', '.join([printer.print_name(v) for v in self.indices + self.variables + self.knowns])
        return args

    def generate_fortran(self, printer: FortranPrinter) -> str:
        args = self.generate_fortran_args(printer)
        code = f'''
        subroutine block_{self.id}_equation({args}, eq)
            use constants
            implicit none
            real(8), intent(out) :: eq(:)'''
        for i in self.indices:
            code += f'''
            integer, intent(in) :: {printer.print_name(i)}'''
        for v in self.variables + self.knowns:
            code += f'''
            real(8), intent(in) :: {printer.print_as_array_arg(v)}'''

        # for v in self.variables + self.knowns:
        #     code += f'''
        #     print *, "range: {printer.print_name(v)}, ", lbound({printer.print_name(v)}), ubound({printer.print_name(v)})'''
        # for v in self.unknowns:
        #     code += f'''
        #     print *, "output: {printer.doprint(v)}, ", {printer.doprint(v)}, F_i'''

        for i, eq in enumerate(self.equations):
            code += f'''
            eq({i+1}) = {printer.doprint(eq.lhs - eq.rhs)} ! {eq.label}'''
        code += f'''
        end subroutine block_{self.id}_equation

        subroutine block_{self.id}_jacobian({args}, jac)
            use constants
            implicit none
            real(8), intent(out) :: jac(:,:)'''
        for i in self.indices:
            code += f'''
            integer, intent(in) :: {printer.print_name(i)}'''
        for v in self.unknowns + self.knowns:
            code += f'''
            real(8), intent(in) :: {printer.print_as_array_arg(v)}'''
        for i, eq in enumerate(self.equations):
            for j, v in enumerate(self.unknowns):
                if v in self.jacobian[eq]:
                    code += f'''
            jac({i+1}, {j+1}) = {printer.doprint(self.jacobian[eq][v])}'''

        code += f'''
        end subroutine block_{self.id}_jacobian
        '''
        return textwrap.dedent(code)

    def __repr__(self) -> str:
        return f'Block #{self.id}(equations={tuple(eq.label for eq in self.equations)}, variables={self.variables}, unknowns={self.unknowns}), knowns={self.knowns}, indices={self.indices})'

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
            # self.plot_dependencies(dependencies)

        # self.plot_dependencies()

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

        assert len(matched) == len(self.equations)

        return matched, equations, unknowns

    def block_decomposition(self, dependencies: list[Dependency], inputs: set[Function]) -> list[Block]:
        
        edges: dict[Node, list[Node]] = defaultdict(list)
        inverse_edges: dict[Node, list[Node]] = defaultdict(list)

        output_indexed: dict[Function, Function] = {}

        for dep in dependencies:
            edges[dep.eq].append(dep.var)
            inverse_edges[dep.var].append(dep.eq)
            if dep.matching:
                edges[dep.var].append(dep.eq)
                inverse_edges[dep.eq].append(dep.var)
            output_indexed[dep.var] = dep.var.subs_index(dep.index_mapping)


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
                    if node in self.equations:
                        equations.append(cast(Equation, node))
                        knowns |= set(cast(list[Function], edges.get(node, [])))
                    else:
                        # assert node in self.unknowns
                        variables.append(cast(Function, node))
                block = Block(block_id, 
                              equations, 
                              variables,
                              [output_indexed[v] for v in variables],
                              list(knowns - set(variables)), 
                              list(self.indices))
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

        print(f'Indices: {self.indices}')

        self.index_margin = 5

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

        print(f'Input: {self.input}')

        constants = set(self.constants)

        self.equations: set[Equation] = set()
        # self.unknowns: set[Function] = set()

        for eq in system.equations.values():
            equation = self.system.eval(eq)
            self.equations.update([cast(Equation, equation)]) 
            # self.unknowns.update(unknowns)

        # for dep in self.dependencies:
            # print(dep)

        index_combinations = tuple(dict(combo) for combo in 
                                   itertools.product(*[[(i, i.min), (i, i), (i, i.max)] 
                                   for i in self.indices]))
        print('Index combinations:', index_combinations)
        class Stage:
            def __init__(self, indices: dict[Index, Expr], inputs: list[Function], blocks: list[Block]):
                self.indices = indices
                self.inputs = inputs
                self.blocks = blocks

            def __repr__(self) -> str:
                s = f'Stage(indices={self.indices}, inputs={self.inputs})):\n'
                for block in self.blocks:
                    s += f'  {block}\n'
                return s

        stages: list[Stage] = []

        for index_combo in index_combinations:
            inputs = set()
            for v in self.input:
                if all(index_combo.get(i, None) == mapped 
                       for i, mapped in v.index_mapping().items()):
                    inputs.update([v.general_form()])
                    
            print(f'Input on this combo: {index_combo}, {inputs}')

            dependencies: list[Dependency] = []
            for eq in self.equations:
                unknowns = self.system.dependencies_of(eq) - constants - inputs  #type:ignore
                for v in unknowns:
                    dependencies.append(Dependency(eq, v.general_form(), v.index_mapping()))
                
            # for dep in dependencies:
            #     print(dep)

            max_matching, eq_deps, unk_deps = self.maximum_matching(dependencies)
            # print('match', max_matching)
            self.plot_dependencies(dependencies)

            output: set[Function] = set()
            for dep in dependencies:
                output.update([dep.var.subs_index(dep.index_mapping)])
            print('output:', output)
            
            input_in_output = True
            if index_combo:
                i = list(index_combo.keys())[0]
                for v_in in inputs:
                    if v_in.at(i, i+1) not in output:
                        input_in_output = False
                        break
            print(input_in_output)

            blocks = self.block_decomposition(dependencies, inputs)
            # for block in blocks:
            #     print(block)

            stage = Stage(index_combo, list(inputs), blocks)
            
            stages.append(stage)
            print(stage)

            if input_in_output:
                break

        # print('Stages:')
        # for stage in stages:
        #     print(stage)

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
            real(8), save :: {pn(c)} = 0.0'''
        code += f'''
        end module constants

        module indices'''
        for i in self.indices:
            code += f'''
            integer, save :: {pn(i)} = 0'''
        code += f'''
        end module indices

        module blocks
        contains
        '''

        for block in stages[0].blocks:
            code += textwrap.indent(block.generate_fortran(printer), ' '*12)

        code += f'''
        end module blocks
                                
        subroutine run_solver(log_path, condition, status, message)
            use, intrinsic :: ieee_arithmetic
            use constants
            use blocks
            implicit none
            double precision dnrm2
            external dnrm2

            character(len=*), intent(in) :: log_path
            real(8), dimension({condition_dim}), intent(in) :: condition
            integer, intent(out) :: status
            character(len=100), intent(out) :: message

            integer :: log_unit
            integer :: ios
            
            real(8), dimension({dim_max}) :: eq
            real(8), dimension({dim_max},{dim_max}) :: jac
            real(8), dimension({dim_max}) :: vars
            integer, dimension({dim_max}) :: ipiv
            integer :: info
            real(8) :: residual
            integer :: newton_iter
            real(8) :: tol = 1e-8
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
            real(8) :: {pn(v)} = 0.0'''
        code += '\n'

        for f in self.system.definitions.keys():
            if f.index:
                code += f'''
            real(8), allocatable :: {printer.print_as_array_arg(f)}'''
            else:
                code += f'''
            real(8) :: {pn(f)} = 0.0'''
        code += '\n'

        for n, c in enumerate(self.constants):
            code += f'''
            {pn(c)} = condition({n + 1})'''
        code += '\n'

        for v in self.variables:
            if v.index:
                code += f'''
            allocate({pn(v)}({",".join(f"{p(i.max - i.min + self.index_margin)}" for i in v.index)}))'''
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

            print *, "Running solver with condition: ", condition

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
            args = block.generate_fortran_args(printer)
            code += f'''
            ! ======================== Block {block.id} ========================
            '''
            for i, v in enumerate(block.unknowns):
                if v.index:
                    cond = ' .and. '.join(f'{p(i.min)} .lt. {p(value)}' for i, value in v.index.items())
                    previous = [(i, value - 1) for i, value in v.index.items()] # type:ignore
                    print(block.id, v, previous)
                    code += f'''
            if ({cond}) then
                {printer.doprint(v)} = {printer.doprint(v.general_form().subs(previous))}
            end if'''

            code += f'''

            call block_{block.id}_equation({args}, eq)

            do newton_iter = 1, 10
                jac = 0

                call block_{block.id}_jacobian({args}, jac)

                ! print *, "Block {block.id}"
                ! print *, "eq", eq
                ! print *, "jac", jac

                ! jac * r = eq; eq = r
                call dgesv({block.dim}, 1, jac, {block.dim}, ipiv, eq, {block.dim}, info)
                if (info /= 0) then
                    write(message, '("Block {block.id}, dgesv error: ", i5)') info
                    status = 1
                    return
                end if
                '''
            for i, v in enumerate(block.unknowns):
                code += f'''
                ! print *, "old ", "{pn(v)}", {pn(v)} 
                {printer.doprint(v)} = {printer.doprint(v)} - eq({i+1})
                ! print *, "new ", "{pn(v)}", {pn(v)} 
                '''
            code += f'''

                ! print *, "red", eq

                call block_{block.id}_equation({args}, eq)

                ! print *, "new_eq", eq

                residual = dnrm2({block.dim}, eq, 1)

                if (ieee_is_nan(residual)) then
                    write(message, '("Block {block.id}, nan: ", i5, E10.4)') newton_iter, residual
                    status = 1
                    return
                end if
                if (residual < tol) exit
            end do
            if (residual >= tol) then
                write(message, '("Block {block.id} not converging: ", i5, ", ", E10.4)') newton_iter, residual
                status = 1
                return
            else
                ! print "('  Block {block.id} converged in ', i3, ' iterations, residual = ', E10.4)", newton_iter, residual
            end if
            '''

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
        print('Condition:', condition_)

        lack_constants = [c for c in self.constants if c not in condition_]
        if lack_constants:
            raise ValueError(f'Value of {lack_constants} must be provided in condition')

        condition_values = np.array(
            [ condition_[c] for c in self.constants ] 
            + [  condition_[v] for v in self.input ])

        result = Result(self.system, directory=directory, name=name)

        status, message = self.module.run_solver(result.path, condition_values)
        if status != 0:
            raise RuntimeError(message.decode('utf-8'))

        for c in self.constants:
            result.set_data(c, condition_[c])
        
        with open(os.path.join(result.path, 'log.bin'), 'rb') as f:
            log_data = np.fromfile(f, dtype=np.float64)
            # print('Log data:', log_data)

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

            # result.set_data(self.variables, log_data)
        

        return result

        index = self.indices[0]
        index_start = index.min or 0
        index_end = index.max or 0
        if index_start and index_start in condition_:
            index_start = condition_[index_start]
        if index_end and index_end in condition_:
            index_end = condition_[index_end]
        N = cast(int, index_end) - cast(int, index_start) + 1

        print(f'{index} = {index_start}, ..., {index_end}')

        value_constants = []
        for v in self.constants:
            if v in condition_:
                value_constants.append(condition_[v])
            else:
                raise ValueError(f'Value of {v} must be provided in condition')
            
        value_indices = []
        for v in self.indices:
            value_indices.append(np.arange(index_start, cast(int, index_end) + 1))

        value_variables = []
        for v in self.variables:
            if v in self.input:
                if v in condition_:
                    values = np.empty((N+1,), dtype=float)
                    values[:] = np.nan
                    values[0] = condition_[v]
                    value_variables.append(values)
                else:
                    raise ValueError(f'Value of {v} must be provided in condition')
            else:
                values = np.empty((N+1,), dtype=float)
                values[:] = np.nan
                values[0] = initial_epsilon * np.random.randn()  # Random initial guess
                value_variables.append(values)

        value_definitions = []
        for f, definition in self.definitions.items():
            value_definitions.append(np.empty((N+1,), dtype=float))

        value_unknowns = np.zeros((len(self.unknowns)), dtype=float)
        value_equation = np.zeros((len(self.unknowns)), dtype=float)
        
        value_jacobian_data = np.zeros(len(self.jacobian_rows))
        self.jacobian_initial_generated(*value_constants, value_jacobian_data)

        value_jacobian = sparse.coo_matrix((value_jacobian_data, (self.jacobian_rows, self.jacobian_cols)), 
                                           shape=(len(self.unknowns), len(self.unknowns))).tocsr()
        
        # np.zeros((len(self.unknowns), len(self.unknowns)), dtype=float)

        result = Result(self.system)

        psudo_inverse_warning = False

        for i_ in tqdm(value_indices[0]):

            for values in value_variables:
                if np.isnan(values[i_ + 1]):
                    values[i_ + 1] = values[i_] # + initial_epsilon * np.random.randn()

            for newton_iter in range(0, newton_max_iter + 1):
                try:
                    self.equations_generated(*value_constants,i_,*value_variables, value_equation)
                    # self.jacobian_generated(*value_constants,i_,*value_variables, value_jacobian)
                    self.jacobian_generated(*value_constants,i_,*value_variables, value_jacobian_data)
                    value_jacobian.data = value_jacobian_data

                    # print('vars 0', np.array(value_variables)[:,newton_iter])
                    # print('vars 1', np.array(value_variables)[:,newton_iter + 1])
                    # print('equations', value_equation)


                    # with np.printoptions(threshold=np.inf):
                    #     print('jacobian',value_jacobian)

                    # print('jacobian', np.linalg.matrix_rank(value_jacobian), value_jacobian.shape)

                    with warnings.catch_warnings(record=True) as w:
                        # value_residual = linalg.solve(value_jacobian, value_equation)
                        warnings.simplefilter("always", category=sparse.linalg.MatrixRankWarning)
                        value_residual = sparse.linalg.spsolve(value_jacobian, value_equation)

                        if any(issubclass(warning.category, sparse.linalg.MatrixRankWarning) for warning in w):

                            value_jacobian_array = value_jacobian.toarray()
                            if not psudo_inverse_warning:
                                psudo_inverse_warning = True
                                # value_jacobian_array = value_jacobian.toarray()
                                print(f'warning: Singular matrix encountered in Newton method, using pseudo-inverse. rank: {np.linalg.matrix_rank(value_jacobian_array)} / {value_jacobian_array.shape[0]}')
                                
                                # u, s, vh = np.linalg.svd(value_jacobian_array)
                                # print('Singular values:', s)

                                import matplotlib.pyplot as plt
                                plt.imshow(value_jacobian.toarray(), cmap='viridis', interpolation='nearest')
                                plt.colorbar()
                                plt.show()
                            value_residual = linalg.pinv(value_jacobian_array) @ value_equation
                            # value_residual = sparse.linalg.lgmres(value_jacobian, value_equation)[0]
                        
                    value_unknowns -= value_residual

                    self.definitions_generated(*value_constants, i_, *value_variables, *value_definitions)
                    self.setter_generated(*value_constants,i_,*value_variables, value_unknowns)

                    if np.linalg.norm(value_residual) < newton_tol:
                        result.newton_converged_iters.append(newton_iter)
                        break
                    if newton_iter == newton_max_iter:
                        raise ValueError('Newton did not converge in {newton_max_iter} iterations')
                except:
                    raise Exception(f'Exception at index {i_}, iter {newton_iter}')

        result.set_data(self.constants, value_constants)
        result.set_data(self.variables, value_variables, N=N)
        result.set_data(self.definitions.keys(), value_definitions, N=N)

        return result
    
    def compile_and_load(self, source: str, libs: list[str] = []) -> Any:

        generate_path = os.path.join(self.tempdir, 'generated.f90')

        with open(generate_path, 'w') as f:
            f.write(source)
        print(generate_path)

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

                
    def print_jacobian(self):
        J: list[list[Expr]] = []
        for i, label in enumerate(self.equations):
            J_i = []
            for j, var in enumerate(self.unknowns):
                J_i.append(self.jacobian.get(label, {}).get(var, 0))
            J.append(J_i)
        
        self.system.show(sp.Matrix(J), label_str='Jacobian') #type:ignore

    def plot_jacobian(self, ticks = True):
        jacobian_matrix = np.zeros((len(self.equations), len(self.unknowns)), dtype=int)
        for i, label in enumerate(self.equations):
            for j, var in enumerate(self.unknowns):
                J_ij = self.jacobian.get(label, {}).get(var, 0)
                if J_ij:
                    if self.system.dependencies_of(J_ij) & set(self.unknowns):
                        jacobian_matrix[i, j] = 2
                    else:
                        jacobian_matrix[i, j] = 1

        plt.figure(figsize=(5, 5))
        plt.imshow(jacobian_matrix, cmap='Greys', vmin=0, vmax=2, interpolation='none')
        plt.title("Jacobian")
        if ticks:
            plt.xticks(ticks=range(len(self.unknowns)), 
                       labels=[f'${self.system.latex(v)}$' for v in self.unknowns], rotation=90)
            plt.yticks(ticks=range(len(self.equations)), 
                       labels=[f'${label}$' for label in self.equations.keys()])
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        plt.show()

        # print('shape:', jacobian_matrix.shape)
        # print('rank:', np.linalg.matrix_rank(jacobian_matrix))

        jacobian_matrix = jacobian_matrix.transpose()

        import networkx as nx
        import scipy.sparse as sp

        # ----- (0) 係数行列 A を用意 ---------------------------------
        A = sp.csr_matrix(jacobian_matrix, dtype=int)

        m, n = A.shape
        row_nodes = [f"r{i}" for i in range(m)]
        col_nodes = [f"c{j}" for j in range(n)]

        # ----- (1) 二部グラフ ----------------------------------------
        G = nx.DiGraph()
        G.add_nodes_from(row_nodes, bipartite=0)
        G.add_nodes_from(col_nodes, bipartite=1)
        for i, j in zip(*A.nonzero()):
            G.add_edge(row_nodes[i], col_nodes[j])      # 式→変数

        # 最大マッチング
        matching = nx.algorithms.bipartite.maximum_matching(G, top_nodes=row_nodes)

        # ----- (2) Dulmage–Mendelsohn グラフ --------------------------
        DM = nx.DiGraph()
        DM.add_nodes_from(G)
        for u, v in G.edges():
            if u in matching and matching[u] == v:
                DM.add_edge(u, v); DM.add_edge(v, u)    # マッチング辺は両向き
            else:
                DM.add_edge(u, v)                       # 非マッチ辺は片向き

        # ----- (3) 欠損/正則/過多ブロックの分類 -----------------------
        unmatched_rows = {r for r in row_nodes if r not in matching}
        unmatched_cols = {c for c in col_nodes if c not in matching.values()}

        def reachable(starts, graph):
            seen = set(starts); stack = list(starts)
            while stack:
                v = stack.pop()
                for w in graph.successors(v):
                    if w not in seen: seen.add(w); stack.append(w)
            return seen

        under = reachable(unmatched_rows, DM)                  # 欠損行側
        over  = reachable(unmatched_cols, DM.reverse(copy=False))  # 欠損列側
        regular = set(DM.nodes()) - under - over               # 中央正則部

        # ----- (4) 正則部で SCC → トポロジカル並び --------------------
        G_reg = DM.subgraph(regular).copy()
        sccs  = list(nx.strongly_connected_components(G_reg))       # SCC 一覧
        cond  = nx.condensation(G_reg)             # 縮約 DAG
        solve_order = list(nx.topological_sort(cond))               # ブロック順

        # ----- (5) 結果を表示 ----------------------------------------
        print("=== Dulmage–Mendelsohn ブロック ===")
        print(f"欠損行ブロック (rows under‑determined): {sorted(under)}")
        print(f"欠損列ブロック (cols over‑determined) : {sorted(over)}")
        print("\n=== 正則部の SCC ===")
        for k, comp in enumerate(sccs):
            print(f"  Block {k}: {sorted(comp)}")
        print("\n解く順番 (トポロジカル順) :", solve_order)

        # optional: ブロック三角化したパターンを確認
        row_perm = [int(v[1:]) for comp_idx in solve_order
                                for v in sccs[comp_idx] if v.startswith('r')]
        col_perm = [int(v[1:]) for comp_idx in solve_order
                                for v in sccs[comp_idx] if v.startswith('c')]
        P = A[row_perm, :][:, col_perm]
        print("\nBTF パターン:\n", P.toarray())

    def plot_dependencies(self, dependencies: list[Dependency]):
        G = nx.DiGraph()
        edge_colors = []
        edge_labels = {}
        for dep in dependencies:
            u = '$' + dep.eq.label + '$'
            v = '$' + self.system.latex(dep.var) + '$'
            G.add_edge(u, v)
            if dep.matching:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
            if (u, v) in edge_labels:
                edge_labels[(u, v)] += f', {dep.index_mapping}'
            else:
                edge_labels[(u, v)] = str(dep.index_mapping)
        left_nodes = {'$' + eq.label + '$' for eq in self.equations}
        pos = nx.bipartite_layout(G, left_nodes)

        plt.figure(figsize=(5, 5))
        nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_color='black', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.2)
        plt.title('Dependencies Graph')
        plt.show()
