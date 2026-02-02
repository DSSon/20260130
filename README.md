# news_impact_alert

뉴스 기반 시장 영향 감지 및 알림 파이프라인 스캐폴딩입니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell에서는 아래 명령어를 실행하세요.

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD에서는 아래 명령어를 실행하세요.

```bat
.venv\Scripts\activate.bat
```

가상환경 활성화 후 아래 명령어를 실행하세요.

```bash
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
