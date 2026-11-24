
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 결과 저장 폴더
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# 1. 테스트 함수와 Gradient
# ==========================================

def f1(x, y):
    """원형 함수 f(x,y) = x^2 + y^2."""
    return x**2 + y**2


def grad_f1(theta):
    """Gradient = (2x, 2y)."""
    x, y = theta
    return np.array([2*x, 2*y])


def f2(x, y):
    """타원형 함수 f(x,y) = x^2 + 10y^2."""
    return x**2 + 10*y**2


def grad_f2(theta):
    """Gradient = (2x, 20y)."""
    x, y = theta
    return np.array([2*x, 20*y])


# ==========================================
# 2. Vanilla Gradient Descent
# ==========================================

class VanillaGD:
    """현재 Gradient만 사용하는 기본 경사하강법."""

    def __init__(self, lr=0.1):
        """학습률을 설정한다."""
        self.lr = lr

    def step(self, theta, grad):
        """
        theta_new = theta - lr * grad
        """
        return theta - self.lr * grad


# ==========================================
# 3. Momentum Gradient Descent
# ==========================================

class Momentum:
    """이전 Gradient 누적값을 반영하는 경사하강법."""

    def __init__(self, lr=0.1, beta=0.9):
        """학습률, 모멘텀 계수, 누적값을 초기화한다."""
        self.lr = lr
        self.beta = beta
        self.v = None

    def step(self, theta, grad):
        """
        v_t = beta * v_(t-1) + grad(theta_t)
        theta_(t+1) = theta_t - lr * v_t
        """
        if self.v is None:
            self.v = np.zeros_like(theta)

        self.v = self.beta * self.v + grad

        return theta - self.lr * self.v


# ==========================================
# 4. 최적화 반복 및 이동 경로 저장
# ==========================================

def optimize(grad_fn, optimizer, initial, steps=100):
    """
    주어진 시작점에서 최적화를 반복한다.

    반환값:
        shape (steps+1, 2)의 이동 경로
    """
    theta = np.array(initial, dtype=float)
    path = [theta.copy()]

    for _ in range(steps):
        grad = grad_fn(theta)
        theta = optimizer.step(theta, grad)

        # 매 반복의 위치 저장
        path.append(theta.copy())

    return np.array(path)


# ==========================================
# 5. 등고선 위 이동 경로 시각화
# ==========================================

def plot_paths(func, paths, title, filename,
               xlim=(-6, 6), ylim=(-6, 6)):
    """여러 최적화 경로를 하나의 등고선 그래프에 표시한다."""

    x = np.linspace(*xlim, 350)
    y = np.linspace(*ylim, 350)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig, ax = plt.subplots(figsize=(8, 7))

    contours = ax.contour(X, Y, Z, levels=18)
    ax.clabel(contours, fontsize=7)

    for label, path in paths.items():
        ax.plot(
            path[:, 0],
            path[:, 1],
            linestyle="--",
            marker="o",
            markersize=3,
            markevery=max(1, len(path) // 15),
            linewidth=1.7,
            label=label
        )

        # 마지막 위치
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            marker="x",
            s=90
        )

    # 최솟값과 시작점
    ax.scatter(0, 0, marker="*", s=180, label="Minimum")
    first_path = next(iter(paths.values()))
    ax.scatter(
        *first_path[0],
        marker="s",
        s=65,
        label="Start"
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


# ==========================================
# 6. 학습률별 수렴/발산 비교
# ==========================================

def plot_learning_rates():
    """
    f(x,y)=x^2+y^2에서 학습률별 손실을 비교한다.

    lr=0.5 : 한 번에 원점 도달
    lr=1.0 : 부호가 바뀌며 진동, 손실 일정
    lr>1.0 : 발산
    """
    rates = [0.1, 0.5, 1.0, 1.1]

    fig, ax = plt.subplots(figsize=(8, 5))

    for lr in rates:
        path = optimize(
            grad_f1,
            VanillaGD(lr=lr),
            initial=[5, 5],
            steps=35
        )

        losses = f1(path[:, 0], path[:, 1])

        # 0인 손실도 로그 축에 표시할 수 있도록 처리
        ax.semilogy(
            np.maximum(losses, 1e-30),
            label=f"lr={lr}"
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Learning Rate: Convergence vs Divergence")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(
        OUTPUT_DIR / "02_learning_rate_comparison.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)


# ==========================================
# 7. 결과 출력 함수
# ==========================================

def print_result(label, path, func):
    """최종 좌표, 손실, 원점까지 거리를 출력한다."""
    x, y = path[-1]

    loss = func(x, y)
    distance = np.linalg.norm(path[-1])

    print(f"\n[{label}]")
    print(f"Final position: ({x:.8f}, {y:.8f})")
    print(f"Final loss: {loss:.10f}")
    print(f"Distance to origin: {distance:.10f}")


# ==========================================
# 8. 모든 과제 실험 실행
# ==========================================

def main():
    """과제 4번의 수렴 검증과 시각화를 모두 실행한다."""

    # --------------------------------------
    # 실험 1: x^2 + y^2, lr=0.1, 100회
    # --------------------------------------
    initial = [5.0, 5.0]
    lr = 0.1
    steps = 100

    vanilla1 = optimize(
        grad_f1,
        VanillaGD(lr=lr),
        initial,
        steps
    )

    momentum1 = optimize(
        grad_f1,
        Momentum(lr=lr, beta=0.9),
        initial,
        steps
    )

    print("=== Experiment 1: x^2 + y^2 ===")

    print_result("Vanilla GD", vanilla1, f1)
    print_result("Momentum GD", momentum1, f1)

    # 과제 조건: 원점 반경 0.1 이내
    assert np.linalg.norm(vanilla1[-1]) < 0.1
    print("\n100-step convergence test: PASS")

    # 두 최적화 경로를 같은 Figure에 표시
    plot_paths(
        f1,
        {
            "Vanilla GD": vanilla1,
            "Momentum GD": momentum1
        },
        title="Vanilla GD vs Momentum: x^2 + y^2",
        filename="01_circle_comparison.png"
    )

    # --------------------------------------
    # 실험 2: 학습률별 발산 실험
    # --------------------------------------
    print("\n=== Experiment 2: Learning Rates ===")

    plot_learning_rates()

    for rate in [0.1, 0.5, 1.0, 1.1]:
        path = optimize(
            grad_f1,
            VanillaGD(lr=rate),
            initial,
            steps=35
        )
        print(
            f"lr={rate:.1f}, "
            f"Final loss={f1(*path[-1]):.6g}"
        )

    # --------------------------------------
    # 실험 3: x^2 + 10y^2
    # Momentum의 수렴 속도 비교
    # --------------------------------------
    initial2 = [5.0, 2.0]
    lr2 = 0.01

    vanilla2 = optimize(
        grad_f2,
        VanillaGD(lr=lr2),
        initial2,
        steps=100
    )

    momentum2 = optimize(
        grad_f2,
        Momentum(lr=lr2, beta=0.9),
        initial2,
        steps=100
    )

    print("\n=== Experiment 3: x^2 + 10y^2 ===")

    print_result("Vanilla GD", vanilla2, f2)
    print_result("Momentum GD", momentum2, f2)

    plot_paths(
        f2,
        {
            "Vanilla GD": vanilla2,
            "Momentum GD": momentum2
        },
        title="Vanilla vs Momentum: x^2 + 10y^2",
        filename="03_ellipse_comparison.png",
        xlim=(-6, 6),
        ylim=(-3, 3)
    )

    # --------------------------------------
    # 실험 4: 타원형 함수의 지그재그 확인
    # --------------------------------------
    # lr=0.09에서는 Vanilla GD의 y값이
    # 매 반복 -0.8배가 되어 진동하며 수렴한다.
    lr3 = 0.09

    vanilla3 = optimize(
        grad_f2,
        VanillaGD(lr=lr3),
        initial2,
        steps=25
    )

    momentum3 = optimize(
        grad_f2,
        Momentum(lr=lr3, beta=0.9),
        initial2,
        steps=25
    )

    plot_paths(
        f2,
        {
            "Vanilla GD": vanilla3,
            "Momentum GD": momentum3
        },
        title="Oscillation Comparison: lr=0.09",
        filename="04_ellipse_oscillation.png",
        xlim=(-6, 6),
        ylim=(-3, 3)
    )

    print("\n=== All experiments completed ===")
    print(f"Saved figures in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
