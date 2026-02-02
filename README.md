# News Impact Alerting

이 프로젝트는 “뉴스가 주가(또는 수익률)에 미치는 영향”을 학습/추정하고, 영향 점수가 높은 뉴스가 등장하면 알림을 발생시키는 파이프라인을 제공합니다.

## 구성

- **데이터 소스 플러그인**: `NewsCollector`, `PriceCollector` 인터페이스를 기반으로 CSV 수집기를 제공합니다.
- **학습/백테스트**: 과거 뉴스와 가격 데이터를 정렬하여 모델을 학습하고, 백테스트 리포트를 생성합니다.
- **실시간 알림**: 최신 뉴스에 대해 영향 점수를 산출하고 임계치 초과 시 알림 이벤트를 발생시킵니다.

## 빠른 시작

### 1) 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) 백테스트 실행 및 모델 저장

```bash
python -m news_impact.cli backtest
```

- 결과 리포트는 `reports/summary.txt`와 `reports/strategy_returns.png`로 저장됩니다.
- 모델은 `artifacts/model.pkl`로 저장됩니다.

### 3) 실시간 뉴스 모니터링

```bash
python -m news_impact.cli monitor
```

- `ALERT_THRESHOLD` 이상 점수를 받은 뉴스는 콘솔에 `[ALERT]` 로그로 출력됩니다.

## 데이터 형식

### 뉴스 CSV (기본: `data/news.csv`)

| column | 설명 |
| --- | --- |
| published_at | ISO8601 날짜/시간 |
| headline | 뉴스 제목 |
| body | 뉴스 본문 |
| source | 뉴스 소스 |

### 가격 CSV (기본: `data/prices.csv`)

| column | 설명 |
| --- | --- |
| date | ISO8601 날짜 |
| close | 종가 |

## 설정

환경 변수는 `.env`에서 설정합니다.

- `NEWS_CSV`: 뉴스 CSV 경로
- `PRICE_CSV`: 가격 CSV 경로
- `MODEL_PATH`: 모델 저장 경로
- `REPORT_DIR`: 리포트 출력 디렉토리
- `ALERT_THRESHOLD`: 알림 임계치

## 테스트

```bash
pytest
```
