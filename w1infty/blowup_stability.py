from mpi4py import MPI
import argparse
import ufl, time, os
import basix.ufl
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from ufl import TrialFunctions, TestFunctions, inner, grad, cross
from initial_data import m_ini

_argument_parser = argparse.ArgumentParser(description="Run the gradient experiment on a quadrilateral mesh.")
_argument_parser.add_argument("--mesh", type=int, default=16, help="Number of cells per coordinate direction (16 or 32 for the paper).")
_cli_args, _petsc_args = _argument_parser.parse_known_args()
if _cli_args.mesh <= 0:
    _argument_parser.error("--mesh must be a positive integer")

N_mesh = _cli_args.mesh; order = 1; T = 1; num_steps_list = [1000]

def compute_energy(m):
    energy_form = 0.5 * inner(grad(m), grad(m)) * dx_lumped
    energy = fem.assemble_scalar(fem.form(energy_form))
    energy = m.function_space.mesh.comm.allreduce(energy, op=MPI.SUM)
    return energy

def compute_w1inf_norm(m_n):
    """计算 ||∇m||_∞, 在 DG-1 的 4 个顶点（与 Q1 有限元节点重合）处求值.
    Q1 梯度在单元间不连续, 相邻公共节点处各单元独立取值, 取全局最大."""
    grad_m_infty = fem.Function(V_infty)
    grad_expr = ufl.sqrt(inner(grad(m_n), grad(m_n)))
    expr = fem.Expression(grad_expr, V_infty.element.interpolation_points)
    grad_m_infty.interpolate(expr)
    local_max = np.max(grad_m_infty.x.array)
    global_max = MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)
    return global_max

def llg_with_history(num_steps):
    dt = T / num_steps
    mho = fem.Function(V)
    mho.interpolate(lambda x: m_ini(x, b= 1.5 * np.pi))

    # 初始化记录数组
    time_history = np.zeros(num_steps + 1)
    energy_history = np.zeros(num_steps + 1)
    w1inf_history = np.zeros(num_steps + 1)

    # 记录初始状态
    time_history[0] = 0.0
    energy_history[0] = compute_energy(mho)
    w1inf_history[0] = compute_w1inf_norm(mho)

    solution_history = []

    # ====== 混合有限元系统 (𝐦̃, 𝐰), 𝐰 = Δ̃_h 𝐦̃ ======
    cell_family = "P" if domain.topology.cell_type == mesh.CellType.triangle else "Q"
    el_V = basix.ufl.element(cell_family, domain.basix_cell(), order, shape=(domain.geometry.dim + 1,))
    el_mixed = basix.ufl.mixed_element([el_V, el_V])
    W = fem.functionspace(domain, el_mixed)

    (m_tilde, w) = TrialFunctions(W)
    (vh, wh) = TestFunctions(W)

    L = (1. / dt) * inner(mho, vh) * dx_lumped

    a = (1. / dt) * inner(m_tilde, vh) * dx_lumped \
        + inner(cross(mho, cross(mho, w)), vh) * dx_lumped \
        + inner(w, wh) * dx_lumped \
        + inner(grad(m_tilde), grad(wh)) * dx_lumped
    solver_options = {
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 2000,
        # "ksp_monitor": None,
        # "ksp_converged_reason": None,
    }
    problem = LinearProblem(a, L, petsc_options=solver_options, petsc_options_prefix="l")

    t = 0.
    global_max_of_max_norm = 0.0
    for k in range(num_steps):
        t += dt
        sol_mixed = problem.solve()

        # 提取 𝐦̃ 分量
        mho.interpolate(sol_mixed.sub(0))

        m_tilde_array = mho.x.array
        norms = np.sqrt(m_tilde_array[0::3] ** 2 + m_tilde_array[1::3] ** 2 + m_tilde_array[2::3] ** 2)

        local_min_norm = np.min(norms)
        global_min_norm = MPI.COMM_WORLD.allreduce(local_min_norm, op=MPI.MIN)
        local_max_norm = np.max(norms)
        global_max_norm = MPI.COMM_WORLD.allreduce(local_max_norm, op=MPI.MAX)

        # 投影到单位球面
        norms = np.where(norms < 1e-12, 1.0, norms)
        mho.x.array[0::3] = m_tilde_array[0::3] / norms
        mho.x.array[1::3] = m_tilde_array[1::3] / norms
        mho.x.array[2::3] = m_tilde_array[2::3] / norms
        mho.x.scatter_forward()

        # 记录当前状态
        time_history[k + 1] = t
        energy_history[k + 1] = compute_energy(mho)
        w1inf_history[k + 1] = compute_w1inf_norm(mho)

        if MPI.COMM_WORLD.rank == 0:
            print(
                f"Step {k + 1}/{num_steps}, t={t:.4f}, E={energy_history[k + 1]:.6e}, ||∇m||_∞={w1inf_history[k + 1]:.6e}",
                f"|m̃|_min={global_min_norm:.6e}, |m̃|_max={global_max_norm:.6e}")

        solution_history.append(mho.copy())

    if MPI.COMM_WORLD.rank == 0:
        max_w1inf = np.max(w1inf_history)
        print(f"  >> max(||∇m||_∞) = {max_w1inf:.6e}")
    return solution_history, time_history, energy_history, w1inf_history

# ============ 主程序 ============
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], [N_mesh, N_mesh], mesh.CellType.quadrilateral)

filename = "mesh_output"
with io.XDMFFile(domain.comm, f"{filename}.xdmf", "w") as file:
    file.write_mesh(domain)

if domain.topology.cell_type == mesh.CellType.triangle:
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": order})
    V = fem.functionspace(domain, ("P", order, (3,)))
else: #quadrilateral
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})
    V = fem.functionspace(domain, ("Q", order, (3,)))

# 创建 W^{1,∞} 计算用的函数空间 (DG-1: 在 Q1 的 4 个顶点处独立求值, 捕获不连续梯度)
V_infty = fem.functionspace(domain, ("DG", 1, (1,)))

global_dofs = V.dofmap.index_map.size_global * domain.geometry.dim
if MPI.COMM_WORLD.rank == 0:
    print(f"Global DOFs: {global_dofs:,}")
    print("-" * 40)

start_time = time.perf_counter()
# 时间步收敛性循环
for k_steps in num_steps_list:
    solutions, time_hist, energy_hist, w1inf_hist = llg_with_history(
        num_steps=k_steps
    )

    if MPI.COMM_WORLD.rank == 0:
        with open(os.path.join(os.path.dirname(__file__), f'energy_{k_steps}_{N_mesh}.txt'), 'w') as f:
            for ti, ei in zip(time_hist, energy_hist):
                f.write(f"{ti:.10e}    {ei:.10e}\n")
        with open(os.path.join(os.path.dirname(__file__), f'w1infty_{k_steps}_{N_mesh}.txt'), 'w') as f:
            for ti, wi in zip(time_hist, w1inf_hist):
                f.write(f"{ti:.10e}    {wi:.10e}\n")

end_time = time.perf_counter()
if MPI.COMM_WORLD.rank == 0:
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")
