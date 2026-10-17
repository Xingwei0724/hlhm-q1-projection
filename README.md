# Harmonic map heat flow: first-order projection FEM

Code for **Unconditional Optimal Error Estimates and Energy Dissipation for a
Linearly Implicit Mass-Lumped Projection Finite Element Method for the Harmonic Map Flow**.

The scripts implement a first-order linearly implicit predictor-projection method
with Cartesian tensor-product Q1 elements and Gauss-Lobatto mass lumping.
The convergence tables use coarse/fine convergence differences.

## Environment

Use Linux or WSL2. `environment.yml` specifies Python 3.11 and DOLFINx 0.10.0.

```bash
micromamba env create -f environment.yml
micromamba activate hlhm-q1-projection
```

## Convergence tables

Run the 2D scripts serially and the 3D scripts with MPI:
```bash
cd converge
python 2d_initial_time.py
python 2d_initial_space.py
mpirun -n 4 python 3d_initial_time.py 
mpirun -n 8 python 3d_initial_space.py
cd ..
```

## Figure 1: discrete energy

```bash
cd energy_stable
python blowup_stability.py
cd ..
```

## Figure 2: gradient and energy

```bash
cd w1infty
python blowup_stability.py --mesh 16
python blowup_stability.py --mesh 32
cd ..
```

Run MATLAB in the corresponding experiment directory, after the Python runs.
Plots open in MATLAB; export them manually if needed.

