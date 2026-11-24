"""SVD를 활용한 64x64 흑백 이미지 압축 및 복원 (AI 심화 학습 과제).

실행: python svd_compression.py
다른 이미지: python svd_compression.py --image /path/to/image.png

수학: A = U @ diag(s) @ Vt
절단 SVD: A_k = U[:, :k] @ diag(s[:k]) @ Vt[:k, :]
단, k_effective = min(k_requested, min(A.shape)).
64x64 행렬의 특이값은 최대 64개이므로 요청 k=100은 실제 k=64로 계산.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "inputs" / "grayscale_64x64.png"
OUTPUT_DIR = BASE / "outputs"
REQUESTED_KS = (10, 50, 100)


def read_grayscale(image_path: Path, size: int = 64) -> np.ndarray:
    """이미지를 읽어 [0,1] 범위의 size x size 흑백 NumPy 배열로 만든다."""
    img = np.asarray(imread(str(image_path)), dtype=np.float64)
    if img.ndim == 3:
        if img.shape[2] < 3:
            raise ValueError("RGB 또는 grayscale 이미지가 필요합니다.")
        img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    if img.ndim != 2:
        raise ValueError("이미지는 흑백 또는 RGB 이미지여야 합니다.")
    if img.max() > 1.0:
        img /= 255.0
    img = np.clip(img, 0.0, 1.0)
    rows = np.linspace(0, img.shape[0] - 1, size).round().astype(int)
    cols = np.linspace(0, img.shape[1] - 1, size).round().astype(int)
    return img[np.ix_(rows, cols)]


def svd_decompose(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy SVD로 U, 특이값 s, Vt 반환 (reduced SVD)."""
    return np.linalg.svd(image, full_matrices=False)


def svd_reconstruct(U: np.ndarray, s: np.ndarray, Vt: np.ndarray,
                    k_requested: int) -> tuple[np.ndarray, int]:
    """요청한 k를 차원 이하로 제한해 상위 특이값으로 원본 행렬 근사."""
    if k_requested < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    k_used = min(k_requested, len(s))
    result = (U[:, :k_used] * s[:k_used]) @ Vt[:k_used, :]
    return result, k_used


def save_comparison(image: np.ndarray, ks: tuple[int, ...] = REQUESTED_KS,
                    output_dir: Path = OUTPUT_DIR) -> list[dict[str, float | int]]:
    """원본 및 k별 복원 이미지를 한 Figure로 저장하고 오차 표 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)
    U, s, Vt = svd_decompose(image)
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(14.5, 4.2))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (64×64)", fontsize=12)
    axes[0].axis("off")

    total_energy = np.sum(s ** 2)
    results = []
    for ax, k_requested in zip(axes[1:], ks):
        recon, k_used = svd_reconstruct(U, s, Vt, k_requested)
        mse = float(np.mean((image - recon) ** 2))
        energy = float(np.sum(s[:k_used] ** 2) / total_energy * 100)
        row = dict(k_requested=k_requested, k_used=k_used,
                   mse=mse, energy_percent=energy)
        results.append(row)

        ax.imshow(np.clip(recon, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Requested k={k_requested} (used {k_used})\n"
                     f"MSE={mse:.7f} | Energy={energy:.2f}%", fontsize=10)
        ax.axis("off")
        plt.imsave(output_dir / f"svd_k{k_requested}.png",
                   np.clip(recon, 0, 1), cmap="gray", vmin=0, vmax=1)

    plt.imsave(output_dir / "original.png", image, cmap="gray", vmin=0, vmax=1)
    fig.suptitle("SVD image reconstruction: k = 10, 50, 100", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_dir / "svd_comparison_k10_k50_k100.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k_requested", "k_used", "mse", "energy_percent"])
        writer.writeheader()
        writer.writerows(results)

    # 검증: 특이값을 전부 쓰면 원본과 부동소수점 허용오차 내에서 같다.
    restored_all, _ = svd_reconstruct(U, s, Vt, len(s))
    assert np.allclose(image, restored_all, atol=1e-12)
    assert all(results[i]["mse"] >= results[i+1]["mse"] - 1e-14
               for i in range(len(results) - 1))
    return results


def main() -> None:
    """64x64 회색조 이미지의 SVD 압축 실험 및 결과를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_INPUT,
                        help="입력 이미지 경로 (기본: inputs/grayscale_64x64.png)")
    args = parser.parse_args()
    np.random.seed(42)
    image = read_grayscale(args.image)
    data = save_comparison(image)
    for row in data:
        print(f"요청 k={row['k_requested']:>3} / 실제 k={row['k_used']:>2} "
              f"/ MSE={row['mse']:.8f} / 보존 에너지={row['energy_percent']:.4f}%")
    print(f"결과 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
