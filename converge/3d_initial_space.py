from mpi4py import MPI
import ufl, time
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from ufl import TrialFunctions, TestFunctions, inner, grad, cross
from dolfinx.fem import create_interpolation_data
from initial_data import ue

def compute_Linf_L2_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data):
    dx_fine = ufl.Measure("dx", domain=domain_fine)
    max_L2_error = 0.0

    for n in range(len(solutions_coarse)):
        # 将粗网格解插值到细网格
        m_coarse_on_fine = fem.Function(V_fine)
        m_coarse_on_fine.interpolate_nonmatching(solutions_coarse[n], fine_cells, interp_data)
        diff = m_coarse_on_fine - solutions_fine[n]

        error_L2_sq_form = inner(diff, diff) * dx_fine
        error_L2_sq = domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(error_L2_sq_form)), op=MPI.SUM)
        error_L2 = np.sqrt(error_L2_sq)

        if error_L2 > max_L2_error: max_L2_error = error_L2
    return max_L2_error

def compute_L2H1_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data):
    dx_fine = ufl.Measure("dx", domain=domain_fine)
    L2H1_error_sq = 0.0

    for n in range(len(solutions_coarse)):
        m_coarse_on_fine = fem.Function(V_fine)
        m_coarse_on_fine.interpolate_nonmatching(solutions_coarse[n], fine_cells, interp_data)
        diff = m_coarse_on_fine - solutions_fine[n]

        error_H1_sq_form = inner(diff, diff) * dx_fine + inner(grad(diff), grad(diff)) * dx_fine
        error_H1_sq = domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(error_H1_sq_form)), op=MPI.SUM)

        L2H1_error_sq += dt * error_H1_sq
    return np.sqrt(L2H1_error_sq)

def compute_discrete_H1semi_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data, dt):
    """计算 sqrt( Δt Σ ||∇ẽ||²_h ), 使用离散内积 (mass lumped)."""
    if domain_fine.topology.cell_type == mesh.CellType.tetrahedron:
        dx_lumped = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": order})
    else:
        dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})

    error_sq = 0.0
    for n in range(len(solutions_coarse)):
        m_coarse_on_fine = fem.Function(V_fine)
        m_coarse_on_fine.interpolate_nonmatching(solutions_coarse[n], fine_cells, interp_data)
        diff = m_coarse_on_fine - solutions_fine[n]

        grad_sq_form = inner(grad(diff), grad(diff)) * dx_lumped
        grad_sq = domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(grad_sq_form)), op=MPI.SUM)
        error_sq += dt * grad_sq

    return np.sqrt(error_sq)

def compute_superconvergent_H1semi_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data, dt):
    """计算 sqrt( Δt Σ ||∇ẽ||² ), 在 超收敛点 (Gauss 求积) 评估梯度.
    Q1: 1-pt Gauss (单元中心, O(h²))
    Q2: 2×2×2 Gauss (±1/√3, O(h³))
    """
    dx_super = ufl.dx(metadata={"quadrature_rule": "default", "quadrature_degree": 2 * order - 2})

    error_sq = 0.0
    for n in range(len(solutions_coarse)):
        m_coarse_on_fine = fem.Function(V_fine)
        m_coarse_on_fine.interpolate_nonmatching(solutions_coarse[n], fine_cells, interp_data)
        diff = m_coarse_on_fine - solutions_fine[n]

        grad_sq_form = inner(grad(diff), grad(diff)) * dx_super
        grad_sq = domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(grad_sq_form)), op=MPI.SUM)
        error_sq += dt * grad_sq

    return np.sqrt(error_sq)


def compute_superconvergent_H1semi_error_coarse(solutions_coarse, solutions_fine, V_coarse, domain_coarse, coarse_cells, interp_fine_to_coarse, dt):
    """在粗网格单元中心评估超收敛 H¹-半范数. 细解插值回粗网格,
    确保求积点位于粗网格单元中心（超收敛点）. """
    dx_super = ufl.dx(metadata={"quadrature_rule": "default", "quadrature_degree": 2 * order - 2})

    error_sq = 0.0
    for n in range(len(solutions_coarse)):
        m_fine_on_coarse = fem.Function(V_coarse)
        m_fine_on_coarse.interpolate_nonmatching(solutions_fine[n], coarse_cells, interp_fine_to_coarse)
        diff = m_fine_on_coarse - solutions_coarse[n]

        grad_sq_form = inner(grad(diff), grad(diff)) * dx_super
        grad_sq = domain_coarse.comm.allreduce(fem.assemble_scalar(fem.form(grad_sq_form)), op=MPI.SUM)
        error_sq += dt * grad_sq

    return np.sqrt(error_sq)

def llg_with_history(domain, dt, return_history=False):
    if domain.topology.cell_type == mesh.CellType.tetrahedron:
        dx_lumped = ufl.dx(metadata={"quadrature_rule": "vertex", "quadrature_degree": order})
        cell_family = "P"
    else:
        dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * order - 1})
        cell_family = "Q"

    # 基础向量空间 V
    V = fem.functionspace(domain, (cell_family, order, (domain.geometry.dim,)))

    mho = fem.Function(V); mho.interpolate(ue)

    if return_history:
        solution_history = []
        tilde_history = []  # 存储中间解 m̃（归一化前）

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
        + inner(grad(m_tilde), grad(wh)) * dx_lumped
    #  刚度项用离散内积和连续内积不影响辅助解模长大于1的性质, 分析有区别.
    L = (1. / dt) * inner(mho, vh) * dx_lumped

    solver_options = {
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
        "sub_pc_type": "ilu",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-12,
        "ksp_max_it": 2000,
        "ksp_monitor": None,
        "ksp_converged_reason": None,
    }
    problem = LinearProblem(a, L, petsc_options=solver_options, petsc_options_prefix="l")

    t = 0.
    for k in range(num_steps):
        t += dt
        sol_mixed = problem.solve()

        # 提取 𝐦̃ 分量并转移到 mho
        mho.interpolate(sol_mixed.sub(0))

        # 统计 m_tilde 在各节点模长的最大/最小值
        m_tilde_array = mho.x.array
        norms = np.sqrt(m_tilde_array[0::3] ** 2 + m_tilde_array[1::3] ** 2 + m_tilde_array[2::3] ** 2)
        if domain.comm.rank == 0:
            print(f"  t={t:.4e}, |m_tilde|: max={norms.max():.6e}, min={norms.min():.6e}")

        if return_history:
            tilde_history.append(mho.copy())  # 保存归一化前的中间解

        # 投影到单位球面
        norms = np.where(norms < 1e-12, 1.0, norms)
        mho.x.array[0::3] = m_tilde_array[0::3] / norms
        mho.x.array[1::3] = m_tilde_array[1::3] / norms
        mho.x.array[2::3] = m_tilde_array[2::3] / norms
        mho.x.scatter_forward()

        if return_history:
            solution_history.append(mho.copy())  # 保存投影后的解

    if return_history:
        return solution_history, tilde_history
    else:
        return mho

# ============ 原始数据，不要更改！ ============
N = [6, 12, 24, 48]; order = 1; T = 0.05; num_steps = 10; dt = T / num_steps

errors_L2H1 = []; errors_LinfL2 = []; errors_H1semi = []; errors_super = []; errors_H1semi_tilde_list = []; errors_super_coarse_list = []; start_time = time.perf_counter()
for n_coarse in N:
    n_fine = 2 * n_coarse
    domain_coarse = mesh.create_box(MPI.COMM_WORLD,[np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])],[n_coarse, n_coarse, n_coarse], mesh.CellType.hexahedron)
    domain_fine = mesh.create_box(MPI.COMM_WORLD, [np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])], [n_fine, n_fine, n_fine], mesh.CellType.hexahedron)

    if domain_coarse.topology.cell_type == mesh.CellType.tetrahedron:
        V_coarse = fem.functionspace(domain_coarse, ("P", order, (domain_coarse.geometry.dim,)))
        V_fine = fem.functionspace(domain_fine, ("P", order, (domain_fine.geometry.dim,)))
    else:  # hexahedron
        V_coarse = fem.functionspace(domain_coarse, ("Q", order, (domain_coarse.geometry.dim,)))
        V_fine = fem.functionspace(domain_fine, ("Q", order, (domain_fine.geometry.dim,)))
    # check_stiffness_diagonal(domain_coarse, V_coarse)

    global_dofs_coarse = V_coarse.dofmap.index_map.size_global * domain_coarse.geometry.dim
    global_dofs_fine = V_fine.dofmap.index_map.size_global * domain_fine.geometry.dim
    if MPI.COMM_WORLD.rank == 0:
        print(f"N={n_coarse:2d}: coarse DOFs={global_dofs_coarse:7,} | fine DOFs={global_dofs_fine:7,}")

    solutions_coarse, tilde_coarse = llg_with_history(domain_coarse, dt, return_history=True)
    solutions_fine, tilde_fine = llg_with_history(domain_fine, dt, return_history=True)
    
    fine_cells = np.arange(domain_fine.topology.index_map(3).size_local, dtype=np.int32)
    interp_data = create_interpolation_data(V_fine, V_coarse, fine_cells, padding=1e-6)
    coarse_cells = np.arange(domain_coarse.topology.index_map(3).size_local, dtype=np.int32)
    interp_fine_to_coarse = create_interpolation_data(V_coarse, V_fine, coarse_cells, padding=1e-6)
    
    # 计算误差（投影解）
    error_L2H1 = compute_L2H1_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data)
    error_LinfL2 = compute_Linf_L2_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data)
    error_H1semi = compute_discrete_H1semi_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data, dt)
    error_super = compute_superconvergent_H1semi_error(solutions_coarse, solutions_fine, V_fine, domain_fine, fine_cells, interp_data, dt)
    # 粗网格超收敛（求积点在粗元中心）
    error_super_coarse = compute_superconvergent_H1semi_error_coarse(
        solutions_coarse, solutions_fine, V_coarse, domain_coarse, coarse_cells, interp_fine_to_coarse, dt)
    
    # 计算误差（中间解 m̃，归一化前）
    error_H1semi_tilde = compute_discrete_H1semi_error(tilde_coarse, tilde_fine, V_fine, domain_fine, fine_cells, interp_data, dt)
    
    if domain_fine.comm.rank == 0:
        print(f"cell_type = {domain_coarse.topology.cell_type}")
        print(f"n_coarse={n_coarse}:")
        print(f"  L^2(0,T;H^1) error        = {error_L2H1:.3e}")
        print(f"  L^infty(0,T;L^2) error     = {error_LinfL2:.3e}")
        print(f"  discrete H^1-semi error   = {error_H1semi:.3e}")
        print(f"  → 中间解 H^1-semi error   = {error_H1semi_tilde:.3e}")
        print(f"  superconv. H^1-semi error = {error_super:.3e}")
        print(f"  → 粗网超收敛 H^1-semi    = {error_super_coarse:.3e}")
    
    errors_L2H1.append(error_L2H1); errors_LinfL2.append(error_LinfL2); errors_H1semi.append(error_H1semi); errors_super.append(error_super)
    errors_H1semi_tilde_list.append(error_H1semi_tilde)
    errors_super_coarse_list.append(error_super_coarse)

if MPI.COMM_WORLD.rank == 0:
    print(f"\n✓ 程序总运行时间: {time.perf_counter() - start_time:.2f} 秒", flush=True)
    for i in range(1, len(N)):
        ratio = N[i] / N[i - 1]
        print(f"\nRefinement: {N[i - 1]} -> {N[i]} (ratio={ratio})")

        rate_L2H1 = np.log(errors_L2H1[i - 1] / errors_L2H1[i]) / np.log(ratio)
        rate_LinftyL2 = np.log(errors_LinfL2[i - 1] / errors_LinfL2[i]) / np.log(ratio)
        rate_H1semi = np.log(errors_H1semi[i - 1] / errors_H1semi[i]) / np.log(ratio)
        rate_super = np.log(errors_super[i - 1] / errors_super[i]) / np.log(ratio)
        rate_H1semi_tilde = np.log(errors_H1semi_tilde_list[i - 1] / errors_H1semi_tilde_list[i]) / np.log(ratio)
        rate_super_coarse = np.log(errors_super_coarse_list[i - 1] / errors_super_coarse_list[i]) / np.log(ratio)
        print(f"  L^2(0,T;H^1) rate           = {rate_L2H1:.4f}")
        print(f"  L^infty(0,T;L^2) rate        = {rate_LinftyL2:.4f}")
        print(f"  discrete H^1-semi rate      = {rate_H1semi:.4f}")
        print(f"  → 中间解 H^1-semi rate     = {rate_H1semi_tilde:.4f}")
        print(f"  superconv. H^1-semi rate    = {rate_super:.4f}")
        print(f"  → 粗网超收敛 H^1-semi rate = {rate_super_coarse:.4f}")