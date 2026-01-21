# KcBERT 욕설/폭언 감지 시스템 사용 가이드

## 목차
1. [설치 방법](#1-설치-방법)
2. [기본 사용법](#2-기본-사용법)
3. [고급 사용법](#3-고급-사용법)
4. [설정 커스터마이징](#4-설정-커스터마이징)
5. [문제 해결](#5-문제-해결)

## 1. 설치 방법

### 1.1 자동 설치 (권장)

```powershell
# PowerShell에서 실행
.\install.ps1
```

설치 스크립트가 자동으로 다음을 수행합니다:
- Python 버전 확인
- 가상환경 생성
- 필요한 패키지 설치
- 디렉토리 구조 생성

### 1.2 수동 설치

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate

# 3. pip 업그레이드
python -m pip install --upgrade pip

# 4. 의존성 설치
pip install -r requirements.txt
```

### 1.3 실행 정책 오류 해결 (Windows)

PowerShell 스크립트 실행 시 오류가 발생하면:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 2. 기본 사용법

### 2.1 대화형 실행

가장 쉬운 방법입니다. 메뉴에서 파일을 선택할 수 있습니다.

```powershell
.\run.ps1
```

실행 화면:
```
예제 파일 목록:
  1. normal_call.txt    - 정상 통화 (욕설 없음)
  2. abusive_call.txt   - 욕설 포함 통화 (욕설 다수)
  3. mixed_call.txt     - 혼합 통화 (불만 표현)
  4. complaint_call.txt - 불만 통화 (경미한 불만)
  5. 직접 파일 경로 입력

선택 (1-5): _
```

### 2.2 간편 실행

파일 경로를 직접 지정하여 실행:

```powershell
# 예제 파일 분석
.\run_simple.ps1 data\samples\normal_call.txt

# 사용자 파일 분석
.\run_simple.ps1 C:\my_data\call_recording.txt
```

### 2.3 Python 직접 실행

더 많은 옵션을 사용하려면 Python을 직접 실행:

```bash
# 기본 실행
python main.py --input data/samples/normal_call.txt

# 짧은 옵션
python main.py -i data/samples/abusive_call.txt
```

## 3. 고급 사용법

### 3.1 임계값 조정

욕설 감지 임계값을 조정할 수 있습니다 (0.0 ~ 1.0):

```bash
# 민감하게 감지 (낮은 임계값)
python main.py -i test.txt --threshold 0.3

# 보수적으로 감지 (높은 임계값)
python main.py -i test.txt --threshold 0.7
```

**임계값 가이드**:
- `0.3`: 매우 민감 - 경미한 불만 표현도 감지
- `0.5`: 균형 (기본값) - 권장 설정
- `0.7`: 보수적 - 명확한 욕설만 감지
- `0.9`: 매우 보수적 - 심각한 욕설만 감지

### 3.2 결과 저장 위치 지정

```bash
# 결과를 특정 위치에 저장
python main.py -i test.txt -o results/my_result.json

# 결과 저장 안함
python main.py -i test.txt --no-save
```

### 3.3 커스텀 설정 파일 사용

```bash
# 다른 설정 파일 사용
python main.py -i test.txt -c my_config.yaml
```

### 3.4 Python 코드에서 직접 사용

```python
from src.detector import AbusiveDetector

# 감지기 초기화
detector = AbusiveDetector(
    model_name="beomi/kcbert-base",
    threshold=0.5,
    max_length=512
)

# 단일 텍스트 분석
text = "분석할 통화 내용"
result = detector.predict(text)

print(f"욕설 감지: {result['is_abusive']}")
print(f"공격성 점수: {result['abusive_score']:.3f}")
print(f"신뢰도: {result['confidence']:.3f}")

# 파일 직접 분석
result = detector.predict_file("data/samples/test.txt")

# 배치 처리
texts = ["첫 번째 텍스트", "두 번째 텍스트"]
results = detector.predict_batch(texts)
```

### 3.5 결과 해석

반환되는 결과 구조:

```json
{
  "text": "입력 텍스트",
  "is_abusive": true,           // 욕설 감지 여부
  "confidence": 0.95,            // 모델 신뢰도
  "abusive_score": 0.87,         // 최종 공격성 점수
  "model_score": 0.82,           // KcBERT 모델 점수
  "rule_score": 0.95,            // 규칙 기반 점수
  "threshold": 0.5,              // 사용된 임계값
  "processing_time": 0.234,      // 처리 시간 (초)
  "source_file": "test.txt"      // 원본 파일 (파일 분석 시)
}
```

**점수 의미**:
- `abusive_score`: 최종 공격성 점수 (0~1)
  - 0.0 ~ 0.3: 정상
  - 0.3 ~ 0.5: 경미한 불만
  - 0.5 ~ 0.7: 중간 수준 공격성
  - 0.7 ~ 1.0: 심각한 욕설/폭언

## 4. 설정 커스터마이징

### 4.1 config.yaml 편집

`config.yaml` 파일을 수정하여 시스템 동작을 조정:

```yaml
model:
  name: "beomi/kcbert-base"      # 사용할 모델
  cache_dir: "./models/kcbert"   # 모델 캐시 위치
  max_length: 512                # 최대 토큰 길이

detection:
  threshold: 0.5                 # 기본 임계값
  batch_size: 8                  # 배치 크기

preprocessing:
  remove_special_chars: true     # 특수문자 제거 여부
  normalize_whitespace: true     # 공백 정규화 여부

output:
  save_results: true             # 결과 자동 저장
  results_dir: "./data/results"  # 결과 저장 위치
  format: "json"                 # 출력 형식

logging:
  level: "INFO"                  # 로그 레벨
  show_processing_time: true     # 처리 시간 표시
```

### 4.2 로그 레벨 조정

- `DEBUG`: 모든 상세 정보 출력
- `INFO`: 일반 정보 출력 (기본값)
- `WARNING`: 경고만 출력
- `ERROR`: 오류만 출력

## 5. 문제 해결

### 5.1 일반적인 오류

#### "Python을 찾을 수 없습니다"

**해결 방법**:
1. Python 3.8 이상 설치: https://www.python.org/downloads/
2. 설치 시 "Add Python to PATH" 옵션 체크
3. 설치 후 터미널 재시작

#### "모듈을 찾을 수 없습니다"

**해결 방법**:
```bash
# 가상환경이 활성화되었는지 확인
# (venv) 프롬프트가 표시되어야 함

# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 의존성 재설치
pip install -r requirements.txt
```

#### "transformers 설치 오류"

**해결 방법**:
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 개별 설치 시도
pip install torch
pip install transformers
pip install tokenizers
```

### 5.2 성능 관련

#### "처리 속도가 너무 느립니다"

**해결 방법**:
1. GPU 사용 (CUDA 설치):
   ```bash
   # CUDA 버전 확인 후 PyTorch 재설치
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. 텍스트 길이 줄이기:
   ```yaml
   # config.yaml에서 max_length 조정
   model:
     max_length: 256  # 기본값: 512
   ```

#### "메모리 부족 오류"

**해결 방법**:
1. 배치 크기 줄이기:
   ```yaml
   detection:
     batch_size: 4  # 기본값: 8
   ```

2. 불필요한 프로그램 종료

### 5.3 정확도 개선

#### "오탐(False Positive)이 많습니다"

**해결 방법**:
```bash
# 임계값을 높여서 실행
python main.py -i test.txt --threshold 0.7
```

#### "미탐(False Negative)이 많습니다"

**해결 방법**:
```bash
# 임계값을 낮춰서 실행
python main.py -i test.txt --threshold 0.3
```

#### "특정 욕설이 감지되지 않습니다"

**해결 방법**:
`src/detector.py`의 `abusive_patterns` 리스트에 패턴 추가:

```python
self.abusive_patterns = [
    '시발', '씨발', '병신', # 기존 패턴
    '추가욕설1', '추가욕설2',  # 새로운 패턴
]
```

### 5.4 파일 인코딩 오류

#### "파일을 읽을 수 없습니다"

**해결 방법**:
1. 파일을 UTF-8로 저장
2. 메모장에서:
   - 파일 열기 → 다른 이름으로 저장
   - 인코딩: UTF-8 선택

### 5.5 PowerShell 실행 오류

#### "이 시스템에서 스크립트를 실행할 수 없습니다"

**해결 방법**:
```powershell
# 관리자 권한 없이 실행 가능
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 또는 일회성 실행
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

## 6. 활용 예시

### 6.1 여러 파일 일괄 처리

```powershell
# PowerShell 스크립트
Get-ChildItem ".\data\*.txt" | ForEach-Object {
    python main.py -i $_.FullName
}
```

### 6.2 결과를 CSV로 변환

```python
import json
import csv

# 결과 파일 읽기
with open('data/results/result.json', 'r', encoding='utf-8') as f:
    result = json.load(f)

# CSV로 저장
with open('result.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['텍스트', '욕설여부', '점수', '신뢰도'])
    writer.writerow([
        result['text'],
        result['is_abusive'],
        result['abusive_score'],
        result['confidence']
    ])
```

### 6.3 실시간 모니터링

```python
import time
from src.detector import AbusiveDetector
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CallFileHandler(FileSystemEventHandler):
    def __init__(self, detector):
        self.detector = detector
    
    def on_created(self, event):
        if event.src_path.endswith('.txt'):
            result = self.detector.predict_file(event.src_path)
            if result['is_abusive']:
                print(f"⚠️ 욕설 감지: {event.src_path}")

# 실행
detector = AbusiveDetector()
handler = CallFileHandler(detector)
observer = Observer()
observer.schedule(handler, path='./data/samples', recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

## 7. 추가 정보

### 7.1 성능 벤치마크

테스트 환경: Intel i5, 16GB RAM, SSD

| 텍스트 길이 | 처리 시간 (CPU) | 처리 시간 (GPU) |
|------------|----------------|----------------|
| 짧음 (< 100자) | 0.3초 | 0.1초 |
| 중간 (100-300자) | 0.5초 | 0.2초 |
| 긴 (300-500자) | 0.8초 | 0.3초 |

### 7.2 라이선스 정보

본 프로젝트는 MIT 라이선스를 따릅니다.
KcBERT 모델은 Apache 2.0 라이선스를 따릅니다.

### 7.3 문의 및 지원

- GitHub Issues: 버그 리포트 및 기능 제안
- 이메일: 프로젝트 관리자에게 문의

---

**업데이트 날짜**: 2026-01-21
