import numpy as np
import matplotlib.pyplot as plt

# Варіант 3: вхідні дані
X = np.array([7, 12, 17, 22, 27, 32], dtype=float)
Y = np.array([8, 7, 6, 5, 4, 3], dtype=float)
n = len(X)

# Обчислення проміжних сум
sum_x  = np.sum(X)
sum_y  = np.sum(Y)
sum_x2 = np.sum(X**2)
sum_xy = np.sum(X * Y)

# Складання і розв'язок нормальних рівнянь МНК
A = np.array([[n, sum_x], [sum_x, sum_x2]], dtype=float)
B = np.array([sum_y, sum_xy], dtype=float)
beta = np.linalg.solve(A, B)
beta0, beta1 = beta

print(f'beta0 = {beta0:.4f}')
print(f'beta1 = {beta1:.4f}')
print(f'Рівняння: y = {beta0:.4f} + ({beta1:.4f})*x')

# Оцінка якості
Y_pred = beta0 + beta1 * X
SS_res = np.sum((Y - Y_pred)**2)
SS_tot = np.sum((Y - np.mean(Y))**2)
R2 = 1 - SS_res / SS_tot
print(f'S (сума кв. похибок) = {SS_res:.6f}')
print(f'R2 = {R2:.6f}')

# Побудова графіка
x_line = np.linspace(5, 35, 300)
y_line = beta0 + beta1 * x_line
plt.scatter(X, Y, color='blue', s=80, label='Експериментальні точки')
plt.plot(x_line, y_line, color='red', linewidth=2,
         label=f'y = {beta0:.4f} - {abs(beta1):.4f}*x')
plt.xlabel('X'); plt.ylabel('Y')
plt.title('МНК. Варіант 3'); plt.legend(); plt.grid(True)
plt.show()
