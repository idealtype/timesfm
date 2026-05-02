---
name: april-ml-research
description: >
  April_basis 프로젝트 전용 ML 연구 스킬.
  TimesFM backbone 위에 함수형 분해 디코더를 얹는 연구와 관련된
  코드 작성, 실험 설계, 데이터 준비, 캐시 구조, 파일 저장 규칙을 다룬다.
  새 실험 파일 작성, 캐싱 코드 작성, 모델 구조 변경, 평가 코드 작성 시 반드시 참조할 것.
---

# April — ML Research Skill

## 0. 코딩 일반 원칙

- **가정을 명시한다.** 불확실한 부분은 구현 전에 질문한다. 해석이 여러 개라면 조용히 하나를 고르지 말고 제시한다.
- **최소한의 코드만 작성한다.** 요청하지 않은 기능, 추상화, 유연성, 불가능한 시나리오에 대한 에러 핸들링을 추가하지 않는다.
- **요청된 범위만 수정한다.** 관련 없는 코드, 주석, 포매팅을 건드리지 않는다. 단, 자신의 변경으로 생긴 orphan(미사용 import/변수/함수)은 반드시 제거한다.
- **다단계 작업은 계획을 먼저 제시한다.** 각 단계마다 검증 기준을 명시하고, 기준을 충족한 뒤 다음 단계로 넘어간다.

---

## 1. 프로젝트 개요

**목표**: TimesFM 2.5 (200M) backbone을 frozen encoder로 사용하고,
그 위에 함수형 구조를 직접 강제하는 분해 디코더를 얹어
예측과 동시에 시계열 성분 분해(Trend / Seasonal / Residual)를 제공한다.

**환경**
- Ubuntu, PyTorch
- Backbone: `google/timesfm-2.5-200m-pytorch` (HuggingFace)
- 데이터셋: Monash TSF

---

## 2. 모델 원칙

- Backbone은 항상 **frozen** (`requires_grad=False`)
- 학습 루프에서 backbone `_encode()` **실시간 호출 금지** → 캐시 로드 사용
- horizon 루프마다 모델 재인스턴스화 금지 → `reset_decoders(cfg)` 패턴으로 backbone 재로드 방지
- `load_backbone=False`가 기본값. 추론 시에만 `load_backbone=True`
- 체크포인트 저장 시 **backbone 가중치 반드시 제외**:
  ```python
  {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
  ```
- **Joint training이 기본**. staged training이 필요한 경우 별도 지시.

---

## 3. 정규화 원칙

**모든 연산은 정규화 공간에서 수행한다. 역정규화는 사용하지 않는다.**

- 정규화 방식: 샘플별 RevIN (`revin` from `timesfm.torch.util`)
- 학습 loss / 평가 MSE·MAE / 시각화 출력: 모두 정규화 공간 값 그대로
- 정규화에 필요한 `mu`, `sigma`는 backbone 임베딩 캐시에 함께 저장

---

## 4. 데이터 준비

### 4-1. Backbone 임베딩 캐시 (read-only 재사용)

기존 생성된 backbone 캐시를 그대로 사용한다. 재생성하지 않는다.

```
/home/sia2/project/4.22prophet/timesfm/prophet_ans/data/monash/cache/
  {dataset_name}/
    backbone_emb_c{context_len}_h{horizon}_stride{stride}.pt
      ├ embeddings [N, 1280]   # last-patch embedding
      ├ mu         [N, 1]
      ├ sigma      [N, 1]
      ├ win_starts [N]         # context 시작 인덱스
      ├ col_ids    [N]         # series id
      ├ context_len, horizon, stride, frequency
```

### 4-2. 실험별 보조 캐시 (prepare_cache.py가 생성)

raw_futures, seasonality_mask, fourier_basis 등 실험에 필요한 파생 캐시는
각 실험의 `prepare_cache.py`가 생성한다.

- .tsf 파싱: `basis_dec/data/monash/prepare.py`의 `iter_rows`, `read_meta` 재사용
- future 슬라이스: `series[win_start + context_len : win_start + context_len + H]`
- 정규화: backbone 캐시의 `mu`, `sigma` 재사용

### 4-3. 데이터 분할

- 배치 내 윈도우는 `win_start` 오름차순 정렬 후 temporal split
- default val_split=0.2 (앞 80% train, 뒤 20% val), 데이터셋별 독립 적용
- 전역 정책: `len(series) < context_len + horizon` 인 시리즈 제외

---

## 5. 캐시 구조

```
basis_dec/experiment/{실험명}/dataset_cache/
  {dataset_name}/
    seasonality_mask.pt          # freq + context_len → seasonality 활성화 여부
    fourier_basis_h{H}.pt        # Fourier basis [H, 2*n_terms] per seasonality type
    raw_futures_h{H}.pt          # futures_n [N, H] + valid_mask [N]
    ...                          # 실험에 따라 추가 캐시 가능
```

**원칙**
- `prepare_cache.py`: 생성 담당
- `train.py`: 읽기만, 생성하지 않음
- 캐시가 없으면 `prepare_cache.py` 먼저 실행

---

## 6. 실험 폴더 구조

**실험마다 모델 구조가 달라지므로 `model/` 폴더는 실험 폴더 안에 포함한다.**

```
basis_dec/
  data/monash/
    prepare.py          # TSF 파싱 유틸 (재사용)
    prophet_config.py   # freq 설정 (재사용)
    *.tsf
  experiment/
    {실험명}/
      model/            ← 이 실험 전용 모델 코드
        __init__.py
        decoder_*.py
        decomp_*.py
      prepare_cache.py
      train.py
      zeroshot_eval.py
      CODEX_PROMPTS.md  ← 구현 프롬프트 (선택)
      dataset_cache/
      checkpoints/
      results/
      config.json
      results.json
      benchmark_results.txt
```

---

## 7. 실험 산출물

모든 실험 종료 시 아래 파일이 생성되어야 한다.

| 파일 | 내용 |
|------|------|
| `config.json` | 실험 설정 전체 |
| `results.json` | horizon별 MSE/MAE |
| `checkpoints/{모델명}_h{H}.pt` | 디코더 체크포인트 (backbone 제외) |
| `checkpoints/tfm_rdh_h{H}.pt` | TFM-RDH baseline 체크포인트 |
| `results/training_losses_h{H}.png` | loss curve |
| `results/decomposition_h{H}.png` | 분해 시각화 |
| `horizon_comparison.png` | 모델 비교 요약 |
| `benchmark_results.txt` | 벤치마크 결과 (overwrite) |

---

## 8. 평가 지표

예측 성능(MSE/MAE)은 **normalized space**에서 계산한다.

분해 품질 지표 (필요 시 `_common/eval_decomposition.py` 재사용):

| 지표 | 설명 |
|------|------|
| Energy share | 각 성분의 분산 비율 (분해 붕괴 탐지) |
| Residual whiteness | Ljung-Box p-value + ACF abs sum |
| Cross-component corr | 성분 간 상관 (직교성) |
| Partial MSE | T / T+S / T+S+R 누적 예측 성능 |

---

## 9. 현재 연구 방향

- 디코더는 **함수형 구조를 직접 강제하고 그 계수를 예측하는 방식**
- 세부 함수 형태(Trend/Seasonal 함수 패밀리)는 실험마다 별도 지시
- 데이터셋: Monash
- **디코더 구조, 계수 예측 방식, loss 설계는 실험마다 별도 지시**
- Baseline: `TFMRandomDirectHead` (TimesFM ResidualBlock, random init)
- train.py / zeroshot_eval.py 분리 구조 유지
