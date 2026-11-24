# SVD 이미지 압축 및 복원 — AI 심화 학습 과제

## PDF 과제 요구사항 (SVD 부분만)

- 흑백 이미지를 행렬로 표현하고 SVD로 분해
- `k=10`, `k=50`, `k=100`에 대한 복원 결과를 하나의 Figure에 비교 출력
- 참고 조건에 맞춰 **64×64 grayscale 이미지** 사용
- NumPy로 수학 계산, Matplotlib으로 시각화
- 생성 이미지 `outputs/` 디렉터리의 PNG로 저장

## 실행

```bash
pip install -r requirements.txt
python svd_compression.py
```

다른 이미지를 사용하려면 다음처럼 실행할 수 있습니다. 입력 이미지는 자동으로 64×64 grayscale로 변환합니다.

```bash
python svd_compression.py --image 경로/이미지.png
```

## 핵심 수식

`A = U Σ Vᵀ`,  `Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ`

- `σᵢ`: i번째 특이값 (큰 순서로 정렬)
- `uᵢ vᵢᵀ`: Rank-1 성분
- 큰 특이값에 대응하는 성분부터 남기면 Rank-k 조건에서 픽셀 제곱오차가 최소화됩니다.
- `MSE = mean((원본 - 복원본)²)`
- `Energy (%) = 100 × sum(s[:k]²)/sum(s²)`는 제곱합 에너지 기준입니다.

**중요:** 64×64 이미지에는 특이값이 최대 64개이므로, `k=100` 요청 시 실제 사용 개수는 `min(100, 64)=64`입니다. 수학적으로 진짜 Rank-100 행렬을 만든 것은 아니며, 요청한 숫자 100을 코드에 유지하되 최대 가능한 성분까지 사용하는 방식입니다. 따라서 k=100 결과는 원본과 부동소수점 오차 범위에서 같습니다.

## ZIP 구성

- `svd_compression.py`: SVD 전용 실행 코드
- `inputs/grayscale_64x64.png`: 실험에 사용한 64×64 흑백 이미지
- `outputs/svd_comparison_k10_k50_k100.png`: **과제 제출용 비교 Figure**
- `outputs/original.png`, `svd_k10.png`, `svd_k50.png`, `svd_k100.png`: 개별 결과 이미지
- `outputs/metrics.csv`: k별 실제 사용 특이값 수, MSE, 보존 에너지
- `requirements.txt`: 설치 라이브러리
