import numpy as np
import matplotlib.pyplot as plt


def Coefficient_Matrix(N):
    n = N * N - 2 * N
    MatrixA = np.eye(n, n)

    for rowNo in range(n):
        for columnNo in range(n):
            if columnNo == rowNo:
                MatrixA[rowNo][columnNo] = -4
            elif columnNo == (rowNo + N) or columnNo == (rowNo - N) or columnNo == (rowNo + 1) or columnNo == (rowNo - 1):
                MatrixA[rowNo][columnNo] = 1
            if rowNo % N == 0 and columnNo == rowNo - 1:
                MatrixA[rowNo][columnNo] = 0
            elif rowNo % N == 0 and columnNo == rowNo + 1:
                MatrixA[rowNo][columnNo] = 2
            elif rowNo % N == N - 1 and columnNo == rowNo + 1:
                MatrixA[rowNo][columnNo] = 0
            elif rowNo % N == N - 1 and columnNo == rowNo - 1:
                MatrixA[rowNo][columnNo] = 2
    return MatrixA


def Known_Vector1(N, J_left, J_right):
    n = N*N - 2*N
    Vectorb1 = np.zeros(n)
    for row in range(n):
        if row % N == 0:
            Vectorb1[row] = J_left
        elif row % N == N-1:
            Vectorb1[row] = J_right
    return Vectorb1

def Known_Vector2(N, rho_upper, rho_lower):
    n = N*N - 2*N
    Vectorb2 = np.zeros(n)
    Vectorb2[0:N] = rho_upper
    Vectorb2[-N:n] = rho_lower
    return Vectorb2

def Unknown_Vector(N, j_left, j_right, rho_upper, rho_lower):
    x = 1
    h = x/N  #step size
    D = 5
    J_left = j_left*2*h/D
    J_right = j_right*2*h/D
    A = Coefficient_Matrix(N)
    b1 = Known_Vector1(N, J_left, J_right)
    b2 = Known_Vector2(N, rho_upper, rho_lower)
    vectorX = np.linalg.solve(A, b1-b2)
    initial_concentration = 50
    gridmatrix = np.ones((N, N)) * initial_concentration
    for row in range(N - 2):
        for column in range(N):
            gridmatrix[row + 1,column] = vectorX[row * N + column]
    return gridmatrix, vectorX
#grid,x = Unknown_Vector(N =5, j_left=100, j_right=100, rho_upper=100, rho_lower=50)
#print('complete vector X is ',grid)
#print("only unknown columns of vector x are ", x)

#Task 5.2
def plotting_task(grid, title, rho_upper, rho_lower):
    plt.imshow(grid, origin='lower', extent=[0, 5, 0, 5])
    plt.colorbar(label=f'Concentration = {rho_upper}, {rho_lower}')
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# 1st subtask in 5.2:  concentration should be constant everywhere
grid1, x1b = Unknown_Vector(N=30, j_left=100, j_right=100, rho_upper=100, rho_lower=100)
plotting_task(grid1, 'Constant Concentration', 100, 100)  # plot says not exactly constant but it is close

# 2nd subtask in 5.2: concentration should be linear
grid2, x2 = Unknown_Vector(N=30, j_left=100, j_right=100, rho_upper=100, rho_lower=50)
plotting_task(grid2, 'Linear Concentration', 100, 50)

# 3rd subtask in 5.2: Change the Dirichlet BC to different pairs of concentration:
# {ρupper, ρlower} = {200, 100} mol · m−3, {50, 30} mol · m−3, {..., ...}.
# How can you set up a single simulation to completely describe the behaviour?
grids = []
titles = []
rho_pairs = [(200, 100), (50, 30), (150, 80), (75, 40)]  # Add more pairs as needed
for rho_upp, rho_low in rho_pairs:
    grid, a = Unknown_Vector(N=30, j_left=100, j_right=100, rho_upper=rho_upp, rho_lower=rho_low)
    grids.append(grid)
    titles.append(f'Concentration = {rho_upp}, {rho_low}')
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, (grid, title) in enumerate(zip(grids, titles)):
    ax = axes[i // 2, i % 2]
    ax.imshow(grid, origin='lower', extent=[0, 5, 0, 5])
    ax.set_title(title)
plt.tight_layout()
plt.show()

# Task:5.2(4)
grids = []
titles = []
N = [5, 30]
for i in N:
    grid, a = Unknown_Vector(i, j_left=100, j_right=100, rho_upper=100, rho_lower=100)
    grids.append(grid)
    titles.append(f'Grid Size(N) = {i}')
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  
for i, (grid, title) in enumerate(zip(grids, titles)):
    ax = axes[i % 2] 
    ax.imshow(grid, origin='lower', extent=[0, 5, 0, 5])
    ax.set_title(title)
plt.tight_layout()
plt.show()

# Test 5.2(5):
from scipy.interpolate import interp2d
def compute_error(N):
    grid, a = Unknown_Vector(N, j_left=100, j_right=100, rho_upper=100, rho_lower=100)
    analytical_solution, b = Unknown_Vector(100, j_left=100, j_right=100, rho_upper=100, rho_lower=100)
    # Interpolate analytical solution to match the grid size
    #analytical_solution_resized = scipy.interpolate.interp2d(np.linspace(0, 1, 100), np.linspace(0, 1, 100), analytical_solution)(np.linspace(0, 1, N), np.linspace(0, 1, N))
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    newx = np.linspace(0, 1, N)
    newy = np.linspace(0, 1, N)
    interpolation = interp2d(x, y, analytical_solution)
    new_analytical_solution = interpolation(newx, newy)
    error = np.linalg.norm(grid - new_analytical_solution)
    print(f'error for (N) = {N} is ', error )
    return error
N_values = [5, 10, 20, 40, 60, 80, 100]
errors = [compute_error(N) for N in N_values]
#errors = [4.904296187944848, 4.050834103478189, 3.3225898549121786, 2.3957571332448215, 0.7824712448275297, 0.0]
print(errors)
plt.plot(N_values, errors, marker='x')
plt.xlabel('Number of Nodes (N)')
plt.ylabel('Error')
plt.title('Convergence Plot')
plt.show()


# Task 5.2(7):
rho_pairs = [(100, 50), (150, 80)]
grids = []
titles = []
for rho_upp, rho_low in rho_pairs:
    grid, c = Unknown_Vector(30, j_left=100, j_right=100, rho_upper=rho_upp, rho_lower=rho_low)
    grids.append(grid)
    titles.append(f'Concentration = {rho_upp}, {rho_low}')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for i, (grid, title) in enumerate(zip(grids, titles)):
    ax = axes[i]
    ax.imshow(grid, origin='lower', extent=[0, 5, 0, 5])
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    #ax.colorbar()
plt.tight_layout()
plt.show()


