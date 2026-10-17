# Harmonic map heat flow: first-order projection FEM

Code for **Unconditional Optimal Error Estimates and Energy Dissipation for a
Linearly Implicit Mass-Lumped Projection Finite Element Method for the Harmonic Map Flow**.

The scripts implement a first-order linearly implicit predictor-projection method
with Cartesian tensor-product Q1 elements and Gauss-Lobatto mass lumping.
The convergence tables use coarse/fine self-convergence differences.

## Environment

Use Linux or WSL2. `environment.yml` specifies Python 3.11 and DOLFINx 0.10.0.
MATLAB is needed for the two plotting scripts.

```bash
micromamba env create -f environment.yml
micromamba activate hlhm-q1-projection
```

## Convergence tables

Run the 2D scripts serially and the 3D scripts with MPI:
```bash
cd converge
python 2d_initial_time.py -lksp_error_if_not_converged
python 2d_initial_space.py -lksp_error_if_not_converged
mpirun -n 4 python 3d_initial_time.py -lksp_error_if_not_converged
mpirun -n 8 python 3d_initial_space.py -lksp_error_if_not_converged
cd ..
```

## Figure 1: discrete energy

Defaults: T = 10, a 128 x 128 mesh, and 10, 20, 40, 80, 160 time steps.
```bash
cd energy_stable
python blowup_stability.py -lksp_error_if_not_converged
matlab -r "energy_stable"
cd ..
```

## Figure 2: gradient and energy

Defaults: T = 1, 1000 time steps, and mesh size parameter 16. Run both meshes:
```bash
cd w1infty
python blowup_stability.py --mesh 16 -lksp_error_if_not_converged
python blowup_stability.py --mesh 32 -lksp_error_if_not_converged
matlab -r "w1infty_p"
cd ..
```

The figure scripts generate the text files consumed by the MATLAB plots.
Run MATLAB in the corresponding experiment directory, after the Python runs.
Plots open in MATLAB; export them manually if needed.
The final 3D spatial comparison uses a 96 x 96 x 96 fine mesh.

The full default PDE experiments have not been rerun as part of preparing
this code package.
