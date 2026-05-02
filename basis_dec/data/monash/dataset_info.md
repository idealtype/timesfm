# Monash TSF Collection

## 기본 정보
- 위치: `prophet_ans/data/monash/*.tsf`
- 입력 형식: Monash `.tsf`
- target columns: 행 단위 전체 series id
- 데이터 분할: temporal test split 없음. 전체 시계열을 train window 생성에 사용
- horizons: 96, 192, 336, 720
- validation 사용 여부: 없음

## context_len 정책
- weekly: 256
- monthly / quarterly / yearly: 64
- daily / hourly / half_hourly / 10_minutes / 4_seconds / minutely: 512
- frequency가 없는 파일은 제외
- 각 horizon에서 `len(series) < context_len + horizon`인 series는 backbone train window 생성에서 제외

## missing value 정책
- 파일명에 `_with_missing_values`가 있는 파일은 제외
- 대응하는 `_without_missing_values` 파일이 있으면 해당 파일만 사용
- 파일명에 missing 여부가 없는 파일은 missing이 없는 파일로 간주

## 캐시 구조
하나의 Monash 폴더 안에서 데이터셋 key별 하위 캐시를 사용한다.

```text
prophet_ans/data/monash/
  cache/
    {dataset_key}/
      prophet_decomp_{series_id}.npz
      prophet_model_{series_id}.json
      backbone_emb_c{context_len}_h{horizon}_stride{context_len}.pt
  diagnostics/
    selected_datasets.json
    {dataset_key}/
      raw_summary.json
      fit_metrics.json
```

## 사용법
```bash
python prepare.py --mode raw
python prepare.py --mode prophet --dataset m3_monthly_dataset --n_workers 4
python prepare.py --mode backbone --dataset m3_monthly_dataset --horizon 96
```

`prophet`, `backbone`, `all` 모드는 데이터 규모가 크므로 기본적으로 `--dataset` 지정이 필요하다.
전체 데이터셋 실행 의도가 명확할 때만 `--all_datasets`를 추가한다.

```bash
python prepare.py --mode prophet --all_datasets --n_workers 4
```

## 제외 정책
`python prepare.py --mode raw` 실행 시 `diagnostics/selected_datasets.json`에 선택/제외 목록을 저장한다.
`--dataset`을 붙인 raw 부분 실행은 해당 dataset의 `raw_summary.json`만 갱신하고 전역 manifest는 덮어쓰지 않는다.
