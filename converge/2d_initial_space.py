from mpi4py import MPI
import ufl, time
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from ufl import TrialFunctions, TestFunctions, inner, grad, cross
from dolfinx.fem import create_interpolation_data

def s(x):  return np.sqrt( ( np.sin(ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+1 )**2 + ( np.cos(2*ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+2 )**2 + ( np.sin(ufl.pi*x[1]) )**2 )
def ue(x): return ( np.sin(ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+1 )/s(x), ( np.cos(2*ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+2 )/s(x), np.sin(ufl.pi*x[1]) / s(x)

def llg(domain, return_history=False):
    # ====== 基础向量空间 V ======
    V = fem.functionspace(domain, ("Q", order, (domain.geometry.dim + 1,)))

    # GLL 质量集中积分
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": 2 * (order + 1) - 3})

    # 初值
    mho = fem.Function(V); mho.interpolate(ue)
    
    if return_history:
        sol_history = []

    # ====== 混合函数空间 V × V 用于 (𝐦̃, 𝐰), 𝐰 = Δ̃_h 𝐦̃ ======
    el_V = basix.ufl.element("Q", domain.basix_cell(), order, shape=(domain.geometry.dim + 1,))
    el_mixed = basix.ufl.mixed_element([el_V, el_V])
    W = fem.functionspace(domain, el_mixed)

    # 混合空间的试/测试函数
    (m_tilde, w) = TrialFunctions(W)
    (vh, wh) = TestFunctions(W)

    # ==== 混合系统变分形式 ====
    a = (1. / dt) * inner(m_tilde, vh) * dx_lumped \
        + inner(cross(mho, cross(mho, w)), vh) * dx_lumped \
        + inner(w, wh) * dx_lumped \
        + inner(grad(m_tilde), grad(wh)) * dx_lumped
    L = (1. / dt) * inner(mho, vh) * dx_lumped

    # 混合系统不是 SPD，使用 GMRES + ILU 预处理
    solver_options = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": "1e-10",
        "ksp_atol": "1e-12",
        "ksp_max_it": "500"
    }
    problem = LinearProblem(a, L, petsc_options=solver_options, petsc_options_prefix="l")

    t = 0.
    for k in range(num_steps):
        t += dt
        sol_mixed = problem.solve()

        # 提取 𝐦̃ 分量并用 interpolate 转移到 V（自动处理 DOF 映射）
        mho.interpolate(sol_mixed.sub(0))

        # 投影到单位球面
        m_tilde_array = mho.x.array
        norms = np.sqrt(m_tilde_array[0::3] ** 2 + m_tilde_array[1::3] ** 2 + m_tilde_array[2::3] ** 2)
        # norms = np.where(norms < 1e-12, 1.0, norms)
        mho.x.array[0::3] = m_tilde_array[0::3] / norms
        mho.x.array[1::3] = m_tilde_array[1::3] / norms
        mho.x.array[2::3] = m_tilde_array[2::3] / norms
        mho.x.scatter_forward()
        
        if return_history:
            sol_history.append(mho.copy())
    
    if return_history:
        return sol_history
    return mho
##########原始数据，不要更改！
N=[16, 32, 64, 128]; order = 1;  T = 0.05; num_steps = 10; dt = T/num_steps
errors_final_L2 = []; errors_final_H1 = []; errors_LinfL2 = []; errors_L2H1 = []; start_time = time.perf_counter()
for n_coarse in N:
    n_fine = 2 * n_coarse
    domain_coarse = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.5, 0.5]), np.array([1.5, 1.5])], [n_coarse, n_coarse], mesh.CellType.quadrilateral)
    domain_fine = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.5, 0.5]), np.array([1.5, 1.5])], [n_fine, n_fine], mesh.CellType.quadrilateral)

    V_coarse = fem.functionspace(domain_coarse, ("Q", order, (domain_coarse.geometry.dim+1,)))
    V_fine = fem.functionspace(domain_fine, ("Q", order, (domain_fine.geometry.dim+1,)))

    solutions_coarse = llg(domain_coarse, return_history=True)
    solutions_fine = llg(domain_fine, return_history=True)

    fine_cells = np.arange(domain_fine.topology.index_map(2).size_local, dtype=np.int64)
    interp_data_u = create_interpolation_data(V_fine, V_coarse, fine_cells, padding=1e-6)
    dx1 = ufl.Measure("dx", domain=domain_fine)
    
    # 时间历程误差（空间收敛：粗解→细网插值，每个时间步对齐）
    max_L2_error = 0.0
    L2H1_error_sq = 0.0
    for n in range(len(solutions_coarse)):
        uho_1 = fem.Function(V_fine)
        uho_1.interpolate_nonmatching(solutions_coarse[n], fine_cells, interp_data_u)
        diff = uho_1 - solutions_fine[n]
        
        error_L2_sq = domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(inner(diff, diff) * dx1)), op=MPI.SUM)
        error_L2 = np.sqrt(error_L2_sq)
        if error_L2 > max_L2_error:
            max_L2_error = error_L2
        
        error_H1_sq = error_L2_sq + domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(diff), grad(diff)) * dx1)), op=MPI.SUM)
        L2H1_error_sq += dt * error_H1_sq
    
    error_LinfL2 = max_L2_error
    error_L2H1 = np.sqrt(L2H1_error_sq)
    
    # 最终时刻误差（保持向后兼容）
    uho_10 = solutions_coarse[-1]; uho_2 = solutions_fine[-1]
    uho_1 = fem.Function(V_fine); uho_1.interpolate_nonmatching(uho_10, fine_cells, interp_data_u)
    
    error_form_u = inner(uho_1 - uho_2, uho_1 - uho_2) * dx1
    error_u = np.sqrt(domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(error_form_u)), op=MPI.SUM))
    error_form_u_H1 = error_form_u + inner(grad(uho_1 - uho_2), grad(uho_1 - uho_2)) * dx1
    error_u_H1 = np.sqrt(domain_fine.comm.allreduce(fem.assemble_scalar(fem.form(error_form_u_H1)), op=MPI.SUM))

    if domain_fine.comm.rank == 0:
        print(f"u_error= {error_u:.3e}, u_error_H1= {error_u_H1:.3e}")
        print(f"L^\u221e(0,T;L^2)= {error_LinfL2:.3e},  L^2(0,T;H^1)= {error_L2H1:.3e}")
    errors_final_L2.append(error_u); errors_final_H1.append(error_u_H1)
    errors_LinfL2.append(error_LinfL2); errors_L2H1.append(error_L2H1)
if MPI.COMM_WORLD.rank == 0:
    print(f"\u2713 程序总运行时间: {time.perf_counter() - start_time:.2f} 秒", flush=True)
    for i in range(1, len(N)):
        ratio = N[i] / N[i - 1]
        print(f"\nRefinement: {N[i - 1]} -> {N[i]} (ratio={ratio})")
        print(f"  \u6700\u7ec8\u65f6\u523b L^2 rate       = {np.log(errors_final_L2[i - 1] / errors_final_L2[i]) / np.log(ratio):.4f}")
        print(f"  \u6700\u7ec8\u65f6\u523b H^1 rate       = {np.log(errors_final_H1[i - 1] / errors_final_H1[i]) / np.log(ratio):.4f}")
        print(f"  L^\u221e(0,T;L^2) rate          = {np.log(errors_LinfL2[i - 1] / errors_LinfL2[i]) / np.log(ratio):.4f}")
        print(f"  L^2(0,T;H^1) rate             = {np.log(errors_L2H1[i - 1] / errors_L2H1[i]) / np.log(ratio):.4f}")
