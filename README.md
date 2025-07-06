--

# Simulation of Diffusion using Finite Differences

This project demonstrates the simulation of diffusion processes in a 2D domain using the finite difference method. It is designed for educational purposes, specifically for the MSSM Practical 1, and includes both Python scripts and a Jupyter notebook for interactive exploration.

## Folder Structure

- **FiniteDiff_Diffusion.py**: Main Python script implementing the finite difference method for 2D diffusion, including matrix assembly, boundary conditions, and visualization.
- **test.py**: A nearly identical script to `FiniteDiff_Diffusion.py`, possibly used for testing or as a backup.
- **Diffusion.ipynb**: Jupyter notebook that walks through the theory, implementation, and visualization of the diffusion simulation, including step-by-step tasks and explanations.
- **Practical01_FiniteDifference_Diffusion.pdf**: The practical handout/report describing the mathematical background, discretization, and assignment tasks.
- **Submission_Diffusion.pdf**: The final submission report, likely containing results, discussion, and answers to the practical tasks.
- **PNG Images**: Visual outputs and task illustrations (e.g., `5.2d.png`, `Task 5.2b.png`, etc.) referenced in the notebook and reports.

## Project Overview

The goal is to solve the steady-state diffusion equation in a 2D square domain using the finite difference method. The code constructs the coefficient matrix and known vectors based on Dirichlet and Neumann boundary conditions, then solves the resulting linear system to obtain the concentration profile.

### Key Features

- **Matrix Assembly**: The code builds the finite difference matrix for the interior nodes, handling boundary conditions explicitly.
- **Boundary Conditions**: Supports both Dirichlet (fixed concentration) and Neumann (fixed flux) boundaries.
- **Visualization**: Uses Matplotlib to plot the resulting concentration fields for various boundary conditions and grid sizes.
- **Convergence Analysis**: Includes routines to compare numerical and analytical solutions, and to plot error convergence as the grid is refined.
- **Parameter Studies**: Easily change upper/lower boundary concentrations and grid resolution to study their effects.

## How to Run

### Requirements

- Python 3.x
- numpy
- matplotlib
- scipy
- Jupyter (for the notebook)

Install dependencies with:
```bash
pip install numpy matplotlib scipy jupyter
```

### Running the Notebook

Open and run `Diffusion.ipynb` in Jupyter Notebook or JupyterLab for an interactive, step-by-step exploration of the simulation and results.

### Running the Script

You can run `FiniteDiff_Diffusion.py` or `test.py` directly to generate plots for the different tasks. The scripts are self-contained and will display figures for each scenario described in the practical.

## Tasks Covered

The code and notebook address the following tasks (as per the practical):

1. **Constant Concentration**: Simulate and visualize a case where the concentration is constant throughout the domain.
2. **Linear Concentration**: Simulate a linear concentration profile between two boundaries.
3. **Parameter Sweep**: Run simulations for various pairs of upper/lower boundary concentrations.
4. **Grid Comparison**: Compare solutions on coarse and fine grids.
5. **Convergence Plot**: Analyze and plot the error as the grid is refined.
6. **Optimization**: Discuss how to speed up the simulation for special cases.
7. **Boundary Effects**: Visualize and interpret the effect of different boundary conditions.

## File Descriptions

- **FiniteDiff_Diffusion.py / test.py**: Core functions include:
  - `Coefficient_Matrix(N)`: Builds the finite difference matrix.
  - `Known_Vector1/2(N, ...)`: Constructs the right-hand side for flux and concentration boundaries.
  - `Unknown_Vector(...)`: Solves the system and returns the concentration grid.
  - `plotting_task(...)`: Plots the concentration field.
  - `compute_error(N)`: Computes error vs. a high-resolution reference solution.
- **Diffusion.ipynb**: Contains all code, explanations, and plots for the practical tasks, making it ideal for learning and demonstration.
- **Practical01_FiniteDifference_Diffusion.pdf**: Reference for the mathematical formulation and assignment details.
- **Submission_Diffusion.pdf**: Final report with results and discussion.
- **PNG Images**: Used for visual reference and included in the notebook/report.

## References

- [Simulation-of-Diffusion-using-Finite-Differences GitHub Repository](https://github.com/Sumanthreddy-DE/Simulation-of-Diffusion-using-Finite-Differences)

## Acknowledgements

This project is part of the MSSM 1 practicals at Friedrich-Alexander-Universität Erlangen-Nürnberg, Lehrstuhl für Informatik 10 (Systemsimulation).

---

Feel free to further customize this README to match your specific submission or add more details as needed!
