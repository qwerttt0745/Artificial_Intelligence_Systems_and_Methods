import numpy as np
import matplotlib.pyplot as plt

# ─── Варіант 3: вхідні дані ────────────────────────────────────────────────────
# Ті самі 6 точок, що й у завданні 1
X = np.array([7, 12, 17, 22, 27, 32], dtype=float)
Y = np.array([8,  7,  6,  5,  4,  3], dtype=float)
n = len(X)   # кількість точок = 6

print("=" * 55)
print("  МНК — КВАДРАТИЧНА РЕГРЕСІЯ  |  Варіант 3")
print("=" * 55)
print(f"X = {X}")
print(f"Y = {Y}")
print(f"n = {n}")

# ─── Проміжні суми для нормальних рівнянь (ступінь 2) ─────────────────────────
# Для квадратичної моделі потрібні суми вищих степенів x:
sum_x   = np.sum(X)           # Σx
sum_x2  = np.sum(X**2)        # Σx²
sum_x3  = np.sum(X**3)        # Σx³  ← нова (порівняно з лінійною регресією)
sum_x4  = np.sum(X**4)        # Σx⁴  ← нова
sum_y   = np.sum(Y)           # Σy
sum_xy  = np.sum(X * Y)       # Σxy
sum_x2y = np.sum(X**2 * Y)    # Σx²y ← нова

print("\n--- Проміжні суми ---")
print(f"Σx   = {sum_x}")
print(f"Σx²  = {sum_x2}")
print(f"Σx³  = {sum_x3}")
print(f"Σx⁴  = {sum_x4}")
print(f"Σy   = {sum_y}")
print(f"Σxy  = {sum_xy}")
print(f"Σx²y = {sum_x2y}")

# ─── Матриця нормальних рівнянь (3×3) та вектор правих частин ─────────────────
# Система A·β = B визначає коефіцієнти β₀, β₁, β₂
# A — симетрична матриця Грама, B — вектор правих частин
A = np.array([
    [n,      sum_x,  sum_x2],   # 1-е рівняння: n·β₀  + Σx·β₁  + Σx²·β₂ = Σy
    [sum_x,  sum_x2, sum_x3],   # 2-е рівняння: Σx·β₀ + Σx²·β₁ + Σx³·β₂ = Σxy
    [sum_x2, sum_x3, sum_x4]    # 3-є рівняння: Σx²·β₀+ Σx³·β₁ + Σx⁴·β₂ = Σx²y
], dtype=float)

B = np.array([sum_y, sum_xy, sum_x2y], dtype=float)

print("\n--- Матриця нормальних рівнянь A ---")
print(A)
print(f"\nВектор правих частин B = {B}")

# ─── Розв'язання системи матричним методом ─────────────────────────────────────
# numpy.linalg.solve використовує LU-розклад — чисельно стійкий метод
beta = np.linalg.solve(A, B)
b0, b1, b2 = beta

print("\n--- Результат ---")
print(f"β₀ = {b0:.6f}")
print(f"β₁ = {b1:.6f}")
print(f"β₂ = {b2:.8f}")
print(f"\nРівняння: y = {b0:.4f} + ({b1:.4f})·x + ({b2:.8f})·x²")

# ─── Передбачені значення та оцінка якості ─────────────────────────────────────
Y_pred  = b0 + b1*X + b2*X**2         # модельні значення у вузлах
SS_res  = np.sum((Y - Y_pred)**2)      # сума квадратів залишків (→ 0 = ідеал)
SS_tot  = np.sum((Y - np.mean(Y))**2)  # загальна дисперсія відносно середнього
R2      = 1 - SS_res / SS_tot          # коефіцієнт детермінації ∈ [0; 1]

print(f"\nS  (сума кв. похибок) = {SS_res:.10f}")
print(f"R² (коефіцієнт детерм.) = {R2:.10f}")

# ─── Таблиця відхилень ─────────────────────────────────────────────────────────
print("\n--- Таблиця порівняння ---")
print(f"{'X':>5} {'Y':>6} {'Ŷ':>12} {'Y-Ŷ':>12} {'(Y-Ŷ)²':>14}")
print("-" * 52)
for xi, yi, yp in zip(X, Y, Y_pred):
    print(f"{xi:>5.0f} {yi:>6.1f} {yp:>12.6f} {yi-yp:>12.8f} {(yi-yp)**2:>14.10f}")
print("-" * 52)

# ─── Побудова графіка ───────────────────────────────────────────────────────────
x_line = np.linspace(5, 35, 300)       # рівномірна сітка — 300 точок для плавної кривої
y_line = b0 + b1*x_line + b2*x_line**2 # значення параболи

plt.figure(figsize=(9, 5))
plt.scatter(X, Y, color='blue', s=90, zorder=5, label='Eксп. точки (варіант 3)')
plt.plot(x_line, y_line, color='green', linewidth=2,
         label=f'y={b0:.3f}+({b1:.4f})x+({b2:.6f})x²')

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('МНК — Квадратична регресія. Варіант 3', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
