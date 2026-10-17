from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from ufl import TrialFunctions, TestFunctions, inner, grad, cross

def s(x):  return np.sqrt( ( np.sin(ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+1 )**2 + ( np.cos(2*ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+2 )**2 + ( np.sin(ufl.pi*x[1]) )**2 )
def ue(x): return ( np.sin(ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+1 )/s(x), ( np.cos(2*ufl.pi*x[0])*np.cos(2*ufl.pi*x[1])+2 )/s(x), np.sin(ufl.pi*x[1]) / s(x)

def llg(num_steps, return_history=False):
    t = 0.; dt = T / num_steps

    # 初值
    uho = fem.Function(w); uho.interpolate(ue)
    
    if return_history:
        sol_history = []

    # 构建混合函数空间 w × w
    el_w = basix.ufl.element("Q", domain.basix_cell(), order, shape=(domain.geometry.dim + 1,))
    el_mixed = basix.ufl.mixed_element([el_w, el_w])
    W = fem.functionspace(domain, el_mixed)

    # 混合空间的试/测试函数
    (uh, w_tilde) = TrialFunctions(W)
    (vh, wh) = TestFunctions(W)

    # 混合系统变分形式
    a = (1. / dt) * inner(uh, vh) * dx_lumped \
        + inner(cross(uho, cross(uho, w_tilde)), vh) * dx_lumped \
        + inner(w_tilde, wh) * dx_lumped \
        + inner(grad(uh), grad(wh)) * dx_lumped
    L = (1. / dt) * inner(uho, vh) * dx_lumped

    solver_options = {
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "ksp_rtol": "1e-10",
        "ksp_atol": "1e-12",
        "ksp_max_it": "500"
    }
    problem = LinearProblem(a, L, petsc_options=solver_options, petsc_options_prefix="l")

    for k in range(num_steps):
        t += dt
        sol_mixed = problem.solve()

        # 提取 𝐮̃ 分量并转移到 uho（自动处理 DOF 映射）
        uho.interpolate(sol_mixed.sub(0))

        # 投影到单位球面
        m_tilde_array = uho.x.array
        norms = np.sqrt(m_tilde_array[0::3] ** 2 + m_tilde_array[1::3] ** 2 + m_tilde_array[2::3] ** 2)
        norms = np.where(norms < 1e-12, 1.0, norms)
        uho.x.array[0::3] = m_tilde_array[0::3] / norms
        uho.x.array[1::3] = m_tilde_array[1::3] / norms
        uho.x.array[2::3] = m_tilde_array[2::3] / norms
        uho.x.scatter_forward()
        
        if return_history:
            sol_history.append(uho.copy())
    
    if return_history:
        return sol_history
    return uho
##########原始数据，不要更改！
N = [32, 64, 128, 256];  order = 1; n = 24; T = 0.05
errors_LinfL2 = []; errors_L2H1 = []; errors_final_L2 = []; errors_final_H1 = []
for num_steps_coarse in N:
    num_steps_fine = 2 * num_steps_coarse
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.5, 0.5]), np.array([1.5, 1.5])], [n, n], mesh.CellType.quadrilateral)
    dx_lumped = ufl.dx(metadata={"quadrature_rule": "GLL", "quadrature_degree": order})

    u_el = basix.ufl.element("Q", domain.basix_cell(), order, shape=(3,))
    w = fem.functionspace(domain, u_el)

    solutions_coarse = llg(num_steps_coarse, return_history=True)
    solutions_fine = llg(num_steps_fine, return_history=True)
    
    dt_coarse = T / num_steps_coarse
    dx_fine = ufl.Measure("dx", domain=domain)
    
    # 时间历程误差
    max_L2_error = 0.0
    L2H1_error_sq = 0.0
    for k in range(len(solutions_coarse)):
        # 对齐物理时间: coarse[k] at t=(k+1)*dt_coarse, fine[2*k+1] at same time
        diff = solutions_coarse[k] - solutions_fine[2 * k + 1]
        
        error_L2_sq = domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(diff, diff) * dx_fine)), op=MPI.SUM)
        error_L2 = np.sqrt(error_L2_sq)
        if error_L2 > max_L2_error:
            max_L2_error = error_L2
        
        error_H1_sq = error_L2_sq + domain.comm.allreduce(fem.assemble_scalar(fem.form(inner(grad(diff), grad(diff)) * dx_fine)), op=MPI.SUM)
        L2H1_error_sq += dt_coarse * error_H1_sq
    
    error_LinfL2 = max_L2_error
    error_L2H1 = np.sqrt(L2H1_error_sq)
    
    # 最终时刻误差（保持向后兼容）
    uho_1 = solutions_coarse[-1]
    uho_2 = solutions_fine[-1]
    final_diff = uho_1 - uho_2
    error_form_u = inner(final_diff, final_diff) * ufl.dx
    error_u = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(error_form_u)), op=MPI.SUM))
    error_form_u_H1 = inner(final_diff, final_diff) * ufl.dx + inner(grad(final_diff), grad(final_diff)) * ufl.dx
    error_u_H1 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(error_form_u_H1)), op=MPI.SUM))

    if domain.comm.rank == 0:
        print(f"u_error= {error_u:.3e}, u_error_H1= {error_u_H1:.3e}")
        print(f"L^\u221e(0,T;L^2)= {error_LinfL2:.3e},  L^2(0,T;H^1)= {error_L2H1:.3e}")
    errors_final_L2.append(error_u); errors_final_H1.append(error_u_H1)
    errors_LinfL2.append(error_LinfL2); errors_L2H1.append(error_L2H1)
if MPI.COMM_WORLD.rank == 0:
    print("=\n")
    print("最终时刻收敛阶:")
    for i in range(1, len(N)):
        rate = np.log(errors_final_L2[i - 1] / errors_final_L2[i]) / np.log(2)
        print(f"  u_rates (L^2)        = {rate:.4f}")
    for i in range(1, len(N)):
        rate = np.log(errors_final_H1[i - 1] / errors_final_H1[i]) / np.log(2)
        print(f"  u_H1_rates (H^1)     = {rate:.4f}")
    print("\n时间历程收敛阶:")
    for i in range(1, len(N)):
        rate = np.log(errors_LinfL2[i - 1] / errors_LinfL2[i]) / np.log(2)
        print(f"  L^\\infty(0,T;L^2)   = {rate:.4f}")
    for i in range(1, len(N)):
        rate = np.log(errors_L2H1[i - 1] / errors_L2H1[i]) / np.log(2)
        print(f"  L^2(0,T;H^1)        = {rate:.4f}")
