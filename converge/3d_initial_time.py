from mpi4py import MPI
import ufl, time
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from ufl import TrialFunctions, TestFunctions, inner, grad, cross
from dolfinx.fem import create_interpolation_data
from initial_data import ue

N_mesh = 24; order = 1; T = 0.05; num_steps_list = [16, 32, 64, 128]

def llg_with_history(num_steps, dt, return_history=False):
    mho = fem.Function(V); mho.interpolate(ue)

    if return_history:
        solution_history = []

    # 确定单元族（与 V 一致）
    cell_family = "P" if domain.topology.cell_type == mesh.CellType.tetrahedron else "Q"

    # 构建混合空间 V × V
    el_V = basix.ufl.element(cell_family, domain.basix_cell(), order, shape=(domain.geometry.dim,))
    el_mixed = basix.ufl.mixed_element([el_V, el_V])
    W = fem.functionspace(domain, el_mixed)

    # 混合空间的试/测试函数
    (m_tilde, w) = TrialFunctions(W)
    (vh, wh) = TestFunctions(W)

    # 混合系统变分形式
    #  方程 1: (1/Δt)(𝐦̃, vh)_h + (𝐦^n × (𝐦^n × 𝐰), vh)_h = (1/Δt)(𝐦^n, vh)_h
    #  方程 2: (𝐰, wh)_h + (∇𝐦̃, ∇wh) = 0   （Δ̃ 算子的弱定义）
    a = (1. / dt) * inner(m_tilde, vh) * dx_lumped \
        + inner(cross(mho, cross(mho, w)), vh) * dx_lumped \
        + inner(w, wh) * dx_lumped \
        + inner(grad(m_tilde), grad(wh)) *dx_lumped
    L = (1. / dt) * inner(mho, vh) * dx_lumped

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
    for k in range(num_steps):
        t += dt
        sol_mixed = problem.solve()

        # 提取 𝐦̃ 分量并转移到 mho
        mho.interpolate(sol_mixed.sub(0))

        # 投影到单位球面
        m_tilde_array = mho.x.array
        norms = np.sqrt(m_tilde_array[0::3] ** 2 + m_tilde_array[1::3] ** 2 + m_tilde_array[2::3] ** 2)
        norms = np.where(norms < 1e-12, 1.0, norms)
        mho.x.array[0::3] = m_tilde_array[0::3] / norms
        mho.x.array[1::3] = m_tilde_array[1::3] / norms
        mho.x.array[2::3] = m_tilde_array[2::3] / norms
        mho.x.scatter_forward()

        if return_history:
            solution_history.append(mho.copy())

    if return_history:
        return solution_history
    else:
        return mho

def compute_LinfL2_error_time(solutions_coarse, solutions_fine, domain, time_ratio=2):
    dx = ufl.Measure("dx", domain=domain)
    max_L2_error = 0.0

    len_coarse = len(solutions_coarse); len_fine = len(solutions_fine)
    expected_len_fine = (len_coarse ) * time_ratio
    if len_fine != expected_len_fine:
        raise ValueError(
            f"Time steps mismatch: coarse={len_coarse - 1}, fine={len_fine - 1}, expected ratio={time_ratio}")

    for n in range(len_coarse):
        sol_c = solutions_coarse[n]
        sol_f = solutions_fine[n * time_ratio + (time_ratio - 1)]
        diff = sol_c - sol_f

        error_L2_sq_form = inner(diff, diff) * dx
        error_L2_sq = domain.comm.allreduce(fem.assemble_scalar(fem.form(error_L2_sq_form)), op=MPI.SUM)
        error_L2 = np.sqrt(error_L2_sq)

        if error_L2 > max_L2_error:
            max_L2_error = error_L2

    return max_L2_error

def compute_L2H1_error_time(solutions_coarse, solutions_fine, domain, dt_coarse, time_ratio=2):
    dx = ufl.Measure("dx", domain=domain)
    L2H1_error_sq = 0.0

    len_coarse = len(solutions_coarse); len_fine = len(solutions_fine)

    # 验证时间步比例
    expected_len_fine = (len_coarse) * time_ratio
    if len_fine != expected_len_fine:
        raise ValueError(
            f"Time steps mismatch: coarse={len_coarse - 1}, fine={len_fine - 1}, expected ratio={time_ratio}")

    for n in range(len_coarse):
        sol_c = solutions_coarse[n]
        # coarse[n] at t = (n+1)*dt_coarse, fine[m] at t = (m+1)*dt_fine
        # dt_coarse = time_ratio * dt_fine, so need m = n*time_ratio + (time_ratio - 1)
        sol_f = solutions_fine[n * time_ratio + (time_ratio - 1)]
        diff = sol_c - sol_f

        error_H1_sq_form = inner(diff, diff) * dx + inner(grad(diff), grad(diff)) * dx
        error_H1_sq = domain.comm.allreduce(fem.assemble_scalar(fem.form(error_H1_sq_form)), op=MPI.SUM)

        L2H1_error_sq += dt_coarse * error_H1_sq

    return np.sqrt(L2H1_error_sq)


# ============ 主程序 ============
errors_L2H1 = []; errors_LinfL2 = []; start_time = time.perf_counter()

domain = mesh.create_box(MPI.COMM_WORLD,[np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])], [N_mesh, N_mesh, N_mesh], mesh.CellType.hexahedron)

if domain.topology.cell_type == mesh.CellType.tetrahedron:
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": order})
    V = fem.functionspace(domain, ("P", order, (domain.geometry.dim,)))
else:# hexahedron
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})
    V = fem.functionspace(domain, ("Q", order, (domain.geometry.dim,)))

global_dofs = V.dofmap.index_map.size_global * domain.geometry.dim
if MPI.COMM_WORLD.rank == 0:
    print(f"Global DOFs: {global_dofs:,}")
    print("-" * 40)

# 2. 时间步收敛性循环
for k_steps in num_steps_list:
    dt_coarse = T / k_steps; dt_fine = T / (2 * k_steps)

    if MPI.COMM_WORLD.rank == 0:
        print(f"\nTime Steps: {k_steps} (dt={dt_coarse:.4f}) vs {2 * k_steps} (dt={dt_fine:.4f})")

    solutions_coarse = llg_with_history(num_steps=k_steps, dt=dt_coarse, return_history=True)
    solutions_fine = llg_with_history(num_steps=2 * k_steps, dt=dt_fine, return_history=True)

    # 计算误差 (L2(0,T; H1))
    error_L2H1 = compute_L2H1_error_time(solutions_coarse, solutions_fine, domain, dt_coarse, time_ratio=2)
    error_LinfL2 = compute_LinfL2_error_time(solutions_coarse, solutions_fine, domain, time_ratio=2)

    if MPI.COMM_WORLD.rank == 0:
        print(f"cell_type = {domain.topology.cell_type}")
        print(f"  L^2(0,T;H^1) error = {error_L2H1:.3e}")
        print(f"  L^infty(0,T;L^2) error = {error_LinfL2:.3e}")

    errors_L2H1.append(error_L2H1)
    errors_LinfL2.append(error_LinfL2)

# 3. 收敛阶分析
if MPI.COMM_WORLD.rank == 0:
    print(f"\n✓ 程序总运行时间：{time.perf_counter() - start_time:.2f} 秒", flush=True)

    for i in range(1, len(num_steps_list)):
        ratio = num_steps_list[i] / num_steps_list[i - 1]

        print(f"\nRefinement: {num_steps_list[i - 1]} -> {num_steps_list[i]} (dt ratio={ratio})")

        rate_L2H1 = np.log(errors_L2H1[i - 1] / errors_L2H1[i]) / np.log(ratio)
        rate_LinfL2 = np.log(errors_LinfL2[i - 1] / errors_LinfL2[i]) / np.log(ratio)

        print(f"  L^2(0,T;H^1) rate  = {rate_L2H1:.4f}")
        print(f"  L^infty(0,T;L^2) rate   = {rate_LinfL2:.4f}")