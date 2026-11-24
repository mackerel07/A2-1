import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """함수 f(x) = x²"""
    return x**2

def derivative(f, x, h=1e-5):
    """중심차분법으로 수치 미분"""
    return (f(x+h) - f(x-h)) / (2*h)

numerical = derivative(f, 3)
actual = 6
error = abs(numerical - actual)

print(f"수치 미분: {numerical:.6f}")
print(f"실제 도함수: {actual}")
print(f"오차: {error:.8f}")
print("검증:", error < 1e-4)

import numpy as np
import matplotlib.pyplot as plt

# 2. 등고선 그래프
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

plt.figure(figsize=(7, 7))
contours = plt.contour(X, Y, Z, levels=[4])
plt.clabel(contours)


# 3. 등고선 f(x,y)=4 위에 점 8개 선택
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
px = 2*np.cos(theta)
py = 2*np.sin(theta)

# Gradient = (2x, 2y)
gx = 2*px
gy = 2*py

# Gradient 화살표
plt.quiver(px, py, gx, gy,
           angles="xy", scale_units="xy", scale=8,
           color="red")

plt.scatter(px, py, color="red", s=15)


# 4. Gradient와 등고선 접선의 수직 여부 검증
# 원 x²+y²=4의 접선 방향: (-y, x)
tx = -py
ty = px

dot = gx*tx + gy*ty

print("Gradient와 접선의 내적:", np.round(dot, 10))
print("수직 검증:", np.allclose(dot, 0))

plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour and Gradient")
plt.axis("equal")
plt.grid()
plt.savefig("calculus_contour.png", dpi=150)
plt.show()