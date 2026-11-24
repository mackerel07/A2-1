
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

angle = np.radians(45)
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])

rotated = R @ circle

S = np.array([[2, 0],
              [0, 0.5]])

scaled = S @ circle

k = 2.0

Sh = np.array([[1, k],
              [0, 1]])

sheared = Sh @ circle

# 신발끈 공식
def area(P):
    """꼭짓점 좌표로 다각형의 면적 계산"""
    x, y = P
    return abs(np.dot(x, np.roll(y, -1)) #roll은 원소를 한 칸씩 이동시키는 함수
             - np.dot(y, np.roll(x, -1))) / 2

# 행렬식과 실제 면적비 비교
for name, A, transformed in [
    ("Rotation", R, rotated),
    ("Scaling", S, scaled),
    ("Shear", Sh, sheared)
]:
    det = abs(np.linalg.det(A))
    ratio = area(transformed) / area(circle)
    error = abs(det - ratio) / det * 100

    print(f"{name}: det={det:.4f}, ratio={ratio:.4f}, error={error:.4f}%")
    print("검증:", error < 1)
    
plt.plot(*circle, label="Original")
plt.plot(*rotated, label="Rotated")
plt.plot(*scaled, label="Scaled")
plt.plot(*sheared, label="Sheared")
plt.axis("equal")
plt.grid()
plt.legend()
plt.show()

# Power Iteration
def power_iteration(A, tol=1e-6, max_iter=1000):
    """반복적인 행렬 곱셈으로 지배적 고유쌍 계산"""
    v = np.array([1.0, 0.0])

    for i in range(max_iter):
        new_v = A @ v
        new_v /= np.linalg.norm(new_v) #크기 1로 정규화

        if np.linalg.norm(new_v - v) < tol:
            v = new_v
            break

        v = new_v #고유벡터로 수렴

    eigenvalue = v @ A @ v #numpy에선 v 트렌즈포즈랑 v랑 같음, #레일리몫
    return eigenvalue, v


A = np.array([[4, 1],
              [1, 3]])

value, vector = power_iteration(A)

# NumPy 고유값과 비교
values, vectors = np.linalg.eig(A)
actual = values[np.argmax(np.abs(values))] #절댓값이 큰 numpy 고유값
error = abs(value - actual) / abs(actual) * 100

print("\n[Power Iteration]")
print(f"계산한 고유값: {value:.4f}")
print(f"계산한 고유벡터: {vector}")
print(f"NumPy 고유값: {actual:.4f}")
print(f"오차: {error:.6f}%")
print("검증:", error < 5)


