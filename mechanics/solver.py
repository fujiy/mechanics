from typing import cast, Any, Optional
from collections import defaultdict
import warnings
import numpy as np
import numpy
from scipy import linalg
from scipy import sparse
from tqdm import tqdm
import sympy as sp
import matplotlib.pyplot as plt
from sympy.printing.numpy import SciPyPrinter
import networkx as nx

from mechanics.system import System
from mechanics.symbol import Function, Expr, Index, Equation, Union
from mechanics.util import python_name, name_type, tuple_ish, to_tuple

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
        
class Result:
    def __init__(self, system: System):
        self.system = system

        self.newton_converged_iters = []
        self._dict = {}

    def set_data(self, keys, values, N=None):
        if N:
            self._dict.update({python_name(key.name): value[:N] for key, value in zip(keys, values)})
        else:
            self._dict.update({python_name(key.name): value for key, value in zip(keys, values)})

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

    def __init__(self, eq: Equation, var: Function, index_map: dict[Index, Expr]):
        self.eq = eq
        self.var = var
        self.index_mapping = index_map
        self.matching = False

    def __repr__(self) -> str:
        return f'{self.eq.label} <- {self.var.name}, {self.index_mapping}'
    
class Block:
    def __init__(self, id: int, equations: list[Equation], unknowns: list[Function], knowns: list[Function]):
        self.id = id
        self.equations = tuple(equations)
        self.unknowns = tuple(unknowns)
        self.knowns = tuple(knowns)

        self.jacobian = {}
        for eq in self.equations:
            derivatives = {}
            for v in unknowns:
                derivative = sp.diff(eq, v)
                if derivative != 0:
                    derivatives[v] = derivative
            if derivatives:
                self.jacobian[name] = derivatives

    def __repr__(self) -> str:
        return f'Block #{self.id}(equations={tuple(eq.label for eq in self.equations)}, unknowns={self.unknowns}), knowns={self.knowns})'

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

    def maximum_matching(self) -> tuple[list[Dependency], 
                                        dict[Equation, list[Dependency]],
                                          dict[Function, list[Dependency]]]:

        equations: defaultdict[Equation, list[Dependency]] = defaultdict(list)
        unknowns: defaultdict[Function, list[Dependency]] = defaultdict(list)

        for dependency in self.dependencies:
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
                if unk_dep.eq in searched_eqs: return []

                # print(f'  {dep};  unk_dep: {unk_dep}')
                if unk_dep.matching:
                    eq_matching = True
                    unk_matching = False
                    for more_dep in equations[unk_dep.eq]:
                        assert more_dep.eq == unk_dep.eq
                        if more_dep.var == unk_dep.var: continue
                        if more_dep.var in searched_vars: return []

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
            # self.plot_dependencies()

        matched: list[Dependency] = []
        matched_eqs: set[Equation] = set()
        matched_unks: set[Function] = set()
        for dep in self.dependencies:
            if dep.matching:
                matched.append(dep)
                assert dep.eq not in matched_eqs
                assert dep.var not in matched_unks
                matched_eqs.add(dep.eq)
                matched_unks.add(dep.var)

        assert len(matched) == len(self.equations)

        return matched, equations, unknowns

    def block_decomposition(self) -> list[Block]:
        
        edges: dict[Node, list[Node]] = defaultdict(list)
        inverse_edges: dict[Node, list[Node]] = defaultdict(list)
        for dep in self.dependencies:
            edges[dep.eq].append(dep.var)
            inverse_edges[dep.var].append(dep.eq)
            if dep.matching:
                edges[dep.var].append(dep.eq)
                inverse_edges[dep.eq].append(dep.var)

        visited: dict[Node, int] = {}
        order = 1

        print('Edges:', [(f'{show_node(k)} -> {", ".join(show_node(n) for n in v)}') for k, v in edges.items()])
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
            # print(f'first_dfs: {show_node(node)}')
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
        

        nodess = []
        for node, i in visited_ordered:
            nodess.append(second_dfs(node))

        blocks: list[Block] = []
        block_id = 0
        for nodes in reversed(nodess):
            if nodes:
                equations = []
                unknowns = []
                knowns = set()
                for node in nodes:
                    if node in self.equations:
                        equations.append(node)
                        knowns |= set(edges.get(node, []))
                    else:
                        assert node in self.unknowns
                        unknowns.append(node)
                block = Block(block_id, equations, unknowns, list(knowns - set(unknowns)))
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

        if input is None:
            self.input = tuple(cast(Function, v) for v in system.state_space())
        else:
            self.input = to_tuple(input)

        self.equations: set[Equation] = set()
        self.unknowns: set[Function] = set()
        self.dependencies: list[Dependency] = []

        for eq in system.equations.values():
            equation = self.system.eval(eq)
            self.equations.update([cast(Equation, equation)]) 
            unknowns = self.system.dependencies_of(equation) #type:ignore
            self.unknowns.update(unknowns)
            for v in unknowns:
                self.dependencies.append(Dependency(eq, v.general_form(), v.index_mapping()))

        max_matching, eq_deps, unk_deps = self.maximum_matching()
        print('match', max_matching)

        blocks = self.block_decomposition()

        for block in blocks:
            print(block)

        return

        # Collect unknowns from definitions and equations

        unknowns_: set[Function] = set()

        for f, definition in system.definitions.items():
            unknowns_.update({v for v in self.system.dependencies_of(definition) 
                              if not system.is_constant(v)})
        for eq in system.equations.values():
            unknowns_.update({v for v in self.system.dependencies_of(eq) 
                              if not system.is_constant(v)})
        for v in self.input:
            if v in unknowns_:
                unknowns_.remove(v)

        for v in unknowns_.copy():
            if self.system[v.name] in self.system.definitions:
                unknowns_.remove(v)

        unknowns = set()
        for u in unknowns_:
            already_exixts = False
            # for v in unknowns.copy():
                # if v.is_general_form_of(u): 
                #     already_exixts = True
                #     break
                # if u.is_general_form_of(v): 
                #     unknowns.remove(v)
                #     break
            if not already_exixts: 
                unknowns.add(u)

        self.unknowns = list(unknowns)

        self.system.show(self.unknowns, label_str='Unknowns')
        # print(f'Variables: {self.variables}')

        indices: list[tuple[Index, ...]] = []

        self.equations: dict[str, Expr] = {}
        self.definitions: dict[Function, Expr] = {}

        for eq in system.equations.values():
            equation = eq.lhs - eq.rhs #type:ignore
            self.equations[eq.label] = cast(Expr, self.system.eval(equation))
            indices.append(self.system.free_index_of(equation))

        for f, definition in system.definitions.items():
            if self.system.dependencies_of(definition) & set(self.unknowns):
                self.equations[f.name] = self.system.eval(f - definition) # type:ignore
            else: 
                self.definitions[f] = definition

        self.jacobian = {}
        for name, eq in self.equations.items():
            derivatives = {}
            for v in unknowns:
                derivative = sp.diff(eq, v)
                if derivative != 0:
                    derivatives[v] = derivative
            if derivatives:
                self.jacobian[name] = derivatives

        # print(indices)
        print(f'Equations: {len(self.equations)}, unknowns: {len(unknowns)}, definitions: {len(system.definitions)}')
        
        printer = PythonPrinter()

        args_str = ' '.join([python_name(v.name) + ',' for v in self.constants + self.indices + self.variables])
        constants_str = ' '.join([python_name(v.name) + ',' for v in self.constants])
        definitions_str = ' '.join([python_name(f.name) + ',' for f in self.definitions.keys()])

        definitions_str = f'def definitions_generated({args_str} {definitions_str}):\n'
        for i, (f, definition) in enumerate(self.definitions.items()):
            definitions_str += f'  {printer.doprint(f)} = {printer.doprint(definition)}  # {name} \n'
        definitions_str += f'  pass\n'

        equations_str = f'def equations_generated({args_str} _):\n'
        for i, (name, eq) in enumerate(self.equations.items()):
            equations_str += f'  _[{i}] = {printer.doprint(eq)}  # {name} \n'
        equations_str += f'  pass\n'

        unknowns_indices = {v: i for i, v in enumerate(self.unknowns)}

        variables_set = set(self.variables)
        self.jacobian_rows = []
        self.jacobian_cols = []
        jacobian_str = f'def jacobian_generated({args_str} _):\n'
        jacobian_initial_str = f'def jacobian_initial({constants_str} _):\n'
        n = 0
        for i, (name, derivatives) in enumerate(self.jacobian.items()):
            for v, derivative in derivatives.items():
                j = unknowns_indices[v]
                if derivative == 0:
                    continue
                self.jacobian_rows.append(i)
                self.jacobian_cols.append(j)
                if self.system.dependencies_of(derivative) & variables_set:
                    jacobian_str += f'  _[{n}] = {printer.doprint(derivative)} # (d/d {v.name}) {name} \n'
                else:
                    jacobian_initial_str += f'  _[{n}] = {printer.doprint(derivative)}\n'
                n += 1
        jacobian_str += f'  pass\n'
        jacobian_initial_str += f'  pass\n'

        setter_str = f'def setter_generated({args_str} _):\n'
        for i, unknown in enumerate(self.unknowns):
            setter_str += f'  {printer.doprint(unknown)} = _[{i}]\n'

        print(f'Jacobian: {list(zip(self.jacobian_rows, self.jacobian_cols))}')
        print(definitions_str)
        print(equations_str)
        print(jacobian_str)
        print(jacobian_initial_str)
        print(setter_str)

        exec(definitions_str, globals())
        exec(equations_str, globals())
        exec(jacobian_str, globals())
        exec(jacobian_initial_str, globals())
        exec(setter_str, globals())

        self.definitions_generated = definitions_generated #type:ignore
        self.equations_generated = equations_generated #type:ignore
        self.jacobian_generated = jacobian_generated #type:ignore
        self.jacobian_initial_generated = jacobian_initial #type:ignore
        self.setter_generated = setter_generated #type:ignore

    def run(self, condition: dict[name_type, float], 
            newton_max_iter: int = 100, newton_tol: float = 1e-8,
            initial_epsilon: float = 1e-6) -> Result:
        condition_ = { self.system(name): value for name, value in condition.items()}

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

    def plot_dependencies(self):
        G = nx.DiGraph()
        edge_colors = []
        for dep in self.dependencies:
            G.add_edge(dep.eq.label, dep.var.name)
            if dep.matching:
                edge_colors.append('red')
            else:
                edge_colors.append('black')
        left_nodes = {eq.label for eq in self.equations}
        pos = nx.bipartite_layout(G, left_nodes)

        plt.figure(figsize=(5, 5))
        nx.draw(G, pos, edge_color=edge_colors, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_color='black', arrows=True)
        plt.title('Dependencies Graph')
        plt.show()