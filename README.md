# news_impact_alert

뉴스 기반 시장 영향 감지 및 알림 파이프라인 스캐폴딩입니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

## 데이터 준비

```bash
python -m src.cli prepare-data
```

## 학습

```bash
python -m src.cli train
```

## 백테스트

```bash
python -m src.cli backtest
```

## 알림 데모 실행

```bash
python -m src.cli alert-demo
```
