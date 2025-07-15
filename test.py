import numpy as np
from scipy.sparse import lil_matrix


def generate_full_rank_sparse_integer_system(N, density=0.01, value_range=(-10, 10), random_state=None):
    """
    フルランクなスパース整数連立一次方程式 Ax = b を生成する。
    
    N: 変数（=行数=列数）
    density: 全体に占める非ゼロ要素割合
    value_range: 非ゼロ整数値の範囲（例：(-5, 5)）
    """
    if random_state is not None:
        np.random.seed(random_state)

    A = lil_matrix((N, N), dtype=int)

    # ステップ1: 各行に1つ以上の非ゼロ（対角線に1を置くことでrankを確保）
    for i in range(N):
        val = 0
        while val == 0:
            val = np.random.randint(value_range[0], value_range[1] + 1)
        A[i, i] = val

    # ステップ2: ランダムな位置に追加で非ゼロ整数を入れる（スパース性の調整）
    total_nonzeros_target = int(N * N * density)
    current_nonzeros = N  # すでに対角にN個ある
    while current_nonzeros < total_nonzeros_target:
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        if A[i, j] == 0:
            val = 0
            while val == 0:
                val = np.random.randint(value_range[0], value_range[1] + 1)
            A[i, j] = val
            current_nonzeros += 1

    # bベクトルも整数で
    b = np.random.randint(value_range[0], value_range[1] + 1, size=N)

    return A.tocsr(), b

N = 5
A, b = generate_full_rank_sparse_integer_system(N, density=0.5, value_range=(-5, 5))

print(f"A.shape = {A.shape}, non zero = {A.nnz}, rank = {np.linalg.matrix_rank(A.toarray())}")
print(A.toarray()) 
print(b)

from mechanics import *
import string

vars = string.ascii_lowercase[:N]

system = (
    System()
    .add_variable(' '.join(vars))
)
for i in range(N):
    system.equate(
        '+'.join([f'{A[i, j]}*{vars[j]}' for j in range(N)]),
        b[i],
    )
system.show_all()
solver = system.solver()

solver.run({})