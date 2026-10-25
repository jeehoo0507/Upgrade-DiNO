# Upgrade-DINO: 수치적으로 안정한 Cross-Entropy로 DINO 개선하기

DINO의 학습 불안정성(특히 teacher temperature 민감도) 중 일부가 **cross-entropy의 수치 오류**에서 오는지 확인한다.
기존 CE와 수치 오류를 고친 **Fixed CE** ([Barron, *A Fix for Softmax and Cross-Entropy*, arXiv 2610.31642](https://arxiv.org/abs/2610.31642))를
같은 설정에서 비교한다.

## 실험 (예비, 4 run)

DINO ViT-S/16을 CIFAR-100에서 50에포치, seed 42, RTX A5000 1장.

| Teacher temperature | Standard CE | Fixed CE |
|---|---|---|
| 0.04 | `t0.04_standard` | `t0.04_fixed` |
| 0.10 | `t0.1_standard` | `t0.1_fixed` |

temperature와 loss 구현 외 모든 설정은 동일하다. 같은 seed라 초기 가중치, 데이터 순서, 증강이 4 run에서 같다 (짝 비교).

## 서버 실행 (명령 하나, 터미널 닫아도 됨)

```bash
cd ~ && git clone https://github.com/jeehoo0507/Upgrade-DiNO upgrade-dino && cd upgrade-dino
source $HOME/.local/bin/env && uv sync
export DATA_ROOT=$HOME/dino/data OUTPUT_ROOT=$HOME/dino/outputs GPUS=0 BACKUP_DIR=/shared/users/<me>/dino
setsid nohup bash -c 'cd ~/upgrade-dino && ./smoke_test.sh && ./scripts/run_all.sh' > $HOME/dino.log 2>&1 < /dev/null &
tail -f $HOME/dino.log            # Ctrl+C는 보기만 멈춤
cat results/summary.md            # 결과 (중간에도: uv run python -m dino.summarize)
```
- `smoke_test.sh`: loss 단위 테스트 → 가짜 CIFAR로 4 run × 2에포치 → 요약 → 정확한 재개 확인 → bf16 loss 옵션 (CPU로 3분 안팎). 끝에 `SMOKE OK`.
- 다시 실행하면 끝난 run(`DONE`)은 건너뛰고, 중단된 run은 `checkpoint.pt`에서 이어간다.
- **멈출 때는 반드시 `./scripts/stop.sh`.** `pkill -f run_all.sh`만 하면 내부 반복(xargs)이 살아남아 다음 run을 계속 띄운다.
  한 run 폴더는 `RUNNING.lock`으로 한 프로세스만 학습할 수 있다.
- 동시 실행: GPU 사용률이 낮으면(데이터 증강이 CPU 병목) `P=2 ./scripts/run_all.sh` (run당 GPU 메모리 ~8.5 GB, A5000에 2개).
- 시간: run 하나 대략 3~5시간 (A5000, batch 64, 2 global 224 + 6 local 96 crop), 4 run 순차로 하루 안팎.
  첫 run 로그의 `epoch N done in Xs (ETA ...)`로 실제 시간을 확인할 것.

## Fixed CE가 고치는 것

DINO loss는 soft target CE다: `L = −Σ p log q`, `p = softmax((t − c)/τ_t)`, `q = softmax(s/τ_s)`, `∂L/∂z_S = q − p`.

| | Standard CE (DINO 원본 코드) | Fixed CE (`dino/losses.py`) |
|---|---|---|
| softmax / log_softmax | `F.softmax`, `F.log_softmax` | 최대 logit을 정확히 0으로 고정, 나머지를 `softplus(−m)`로 처리 |
| gradient | autograd: `q·Σp − p` (거의 같은 두 확률의 뺄셈) | `δ = z_S − z_T`에서 직접: `d = δ − Σpδ`, `s = d − log1p(Σ p·expm1(d))`, `u = p·expm1(s)`, `g = u − q·Σu` |
| teacher = student일 때 | gradient가 0이 아님 (fp32에서 ~1e-7) | 정확히 0 |
| 합이 0 | 근사적으로만 | 구성상 정확히 0 |

논문에 없는 보호 장치 2개를 넣었다. τ_t = 0.04에서는 65,536차원 logit의 차이가 수백까지 커져서 `expm1`이 넘칠 수 있기 때문이다.
1. `max|d| ≥ 20`인 행은 같은 값을 주는 `s = d − logsumexp(log p + d)`로 계산한다.
2. `s ≥ 20`인 원소는 `u = q − p`를 직접 쓴다. 이 경우 p는 0으로 사라지고 q는 남아서 뺄셈 상쇄가 없다.

두 경로 모두 실수 연산에서는 같은 값이며, 단위 테스트에서 float64 기준과 비교했다.
단위 테스트 기준 fp32, teacher–student 근접 시 gradient 상대 오차: standard 3.0e-4 → fixed 9.3e-8.

- loss는 두 경우 모두 **float32**로 계산한다 (네트워크 forward만 fp16 autocast, DINO와 같음; `--loss-dtype bf16`으로 변경 가능).
  따라서 두 run의 차이는 CE를 어떻게 계산하느냐 하나뿐이다.

## 지표 (`results/summary.md`, `summary.png`)

| 지표 | 정의 | 파일 |
|---|---|---|
| k-NN accuracy | teacher backbone CLS 특징, 가중 k-NN (k=20, T=0.07, DINO `eval_knn`), 10에포치마다 + 마지막 | `knn.csv` |
| Training loss / spike | 매 iteration loss. spike = 직전 200 step 중앙값보다 6×(1.4826·MAD) 이상 그리고 0.05 이상 큼 | `steps.csv` |
| non-finite / skipped | loss가 NaN·inf인 step, fp16 overflow로 건너뛴 step | `steps.csv` |
| Feature variance | test 특징(L2 정규화)의 차원별 표준편차 평균 × √D (붕괴 시 → 0), 평균 쌍별 cos (붕괴 시 → 1); 매 step teacher 특징 std | `knn.csv`, `steps.csv` |
| Gradient relative error | 50 iteration마다 실제 logit(student view 1 vs teacher view 0)으로 `‖g − g_fp64‖ / ‖g_fp64‖`, **두 구현 × fp32/bf16 모두** 측정 | `graderr.csv` |
| Temperature 민감도 | k-NN(τ_t 0.10) − k-NN(τ_t 0.04), loss별 | summary |

**후속 실험 판단 규칙 (사전 고정, seed 1개라 예비 신호)**: 같은 temperature에서 Fixed CE의 k-NN이 +1.0%p 이상 높거나,
loss spike가 절반 이하로 줄면서 k-NN이 0.5%p보다 나빠지지 않으면 "후속 실험 가치 있음". 그때 seed와 temperature를 늘린다.

## 기본 설정 (`dino/train.py`, DINO `main_dino.py` 기준)

ViT-S/16 (timm, 처음부터 학습, drop path 0.1 / teacher 0), DINO head (out 65,536, hidden 2048, bottleneck 256, 마지막 층 정규화),
2 global crop 224 (scale 0.4–1) + 6 local crop 96 (scale 0.05–0.4), batch 64, AdamW lr 5e-4 × batch/256, warmup 10에포치, cosine → 1e-6,
weight decay 0.04 → 0.4, teacher momentum 0.996 → 1, student temperature 0.1, center momentum 0.9, gradient clip 3.0 (파라미터별),
마지막 층 1에포치 고정, **teacher temperature 고정 (warmup 없음)**. CIFAR 32×32 이미지는 crop할 때 224/96으로 확대한다.
바꾸려면 `EXTRA="--local-crops 10 --batch-size 128" ./scripts/run_all.sh`처럼 넘기면 4 run에 똑같이 적용된다.

## 검수 반영 (외부 코드 리뷰)

| 지적 | 판단 | 반영 |
|---|---|---|
| ① 실패한 step에서도 EMA teacher가 움직임 | 맞음 | optimizer step이 실제로 된 경우(loss 유한 + fp16 overflow skip 아님)에만 EMA. center는 손실 계산 밖으로 빼서 step 뒤에 갱신하고, teacher 출력이 비유한이면 건너뜀. `steps.csv`의 `ema`, `center` 열로 확인. (DINO 원본은 fp16 skip step에도 EMA를 함 → 작은 차이로 명시) |
| ② bf16 gradient 오차에 logit 반올림 손실이 섞임 | 맞음 | `rel_err_*` = **같은 (반올림된) 입력에서의 float64 gradient 대비 = CE 계산 자체의 오차**, `e2e_err_median` = 원래 fp32 logit 대비 (참고). 단위 테스트에서 bf16·근접 시 CE 오차 standard 473% vs fixed 0.68%. 입력 반올림(~0.03)이 teacher–student 차이보다 크면 두 값이 크게 달라서 분리가 필요함을 확인 |
| ③ 재개 시 RNG 미복원 | 맞음 | 매 에포치 시작에 모든 RNG를 (seed, epoch)로 다시 시드 → 에포치 경계에서 재개하면 중단 없는 run과 **완전히 같음**. smoke에서 "에포치 0 후 중단 → 재개"가 모든 step loss와 최종 k-NN까지 일치하는지 검사 |
| ④ τ 0.10의 실패를 수치 오류로 단정 못 함 | 맞음 | summary에 teacher entropy를 같이 두고 해석 주의 문구 추가. 후속 후보: τ_t 0.02 (`EXTRA` 대신 `dino.train --teacher-temp 0.02`) |
| A5000 1 epoch VRAM 확인 | 반영 | 매 에포치 peak GPU 메모리를 로그(`epoch N done ... peak GPU mem`)와 `log.csv`의 `max_mem_gb`에 기록. OOM이면 `EXTRA="--batch-size 32"` |

## 해석할 때 주의

- **fp32 loss에서는 효과가 작을 수 있다 (중요).** DINO head 출력은 정규화된 벡터의 코사인 값(−1~1)이고 fp16으로 계산된다.
  그래서 logit 자체에 이미 ~5e-4 수준의 반올림 잡음이 있다. fp32 CE의 계산 오차(~1e-7, teacher–student 근접 시 ~3e-4)는
  대부분 이보다 작다. 4개 run에서 차이가 없으면 "DINO에서는 수치 오류가 중요하지 않다"가 아니라
  "fp32 loss에서는 오차가 작다"까지만 말할 수 있다. 오차가 큰 조건을 보려면 후속으로 **loss를 bf16으로 계산**한다:
  `EXTRA="--loss-dtype bf16" OUTPUT_ROOT=$HOME/dino/outputs_bf16 ./scripts/run_all.sh`
- 인과: Fixed CE에서 gradient 오차와 spike가 함께 줄어도, DINO 불안정성이 수치 오류 **때문**이라고 확정할 수는 없다.

- seed 1개라 차이가 작으면 노이즈일 수 있다. 판단 규칙을 넘으면 seed를 늘려 재현성부터 확인한다.
- Fixed CE의 이점은 논문 기준으로 **teacher와 student가 가까울 때**(뺄셈 상쇄)와 **logit 차이가 아주 클 때**에 나온다.
  학습 초반처럼 둘이 멀 때는 두 구현의 오차가 비슷하다 (smoke에서도 확인). `graderr.csv`의 시간 변화를 같이 볼 것.
- teacher temperature warmup을 끈 이유: temperature가 이번 실험의 변수라서다. DINO 원본은 0.07 이상에서 warmup을 권장한다.
