# sLLM 기반 욕설/폭언 감지 시스템

## 🎯 개요

KcBERT 외에 sLLM(small Language Model)을 사용한 욕설 감지 방식을 추가로 제공합니다.
온디바이스에서 실행 가능한 4B 파라미터 모델을 사용합니다.

## 🤖 사용 모델

- **모델명**: Midm-2.0-Mini-Instruct
- **크기**: 4B 파라미터
- **형식**: GGUF (Q4_K_M 양자화)
- **파일**: `Midm-2.0-Mini-Instruct-Q4_K_M.gguf`
- **용량**: 약 2.5GB

## 🆚 KcBERT vs sLLM 비교

| 특징 | KcBERT | sLLM |
|------|--------|------|
| **정확도** | 높음 (Fine-tuning 후) | 높음 (문맥 이해) |
| **속도** | 빠름 (0.2-0.3초) | 중간 (1-3초) |
| **문맥 이해** | 제한적 | 우수 |
| **판단 근거** | 없음 | 제공 |
| **온디바이스** | ✅ | ✅ |
| **GPU 필요** | 선택 | 선택 |
| **모델 크기** | 440MB | 2.5GB |
| **Fine-tuning** | 필요 | 불필요 |

## 📥 설치

### 1. llama-cpp-python 설치

```bash
# CPU 버전
pip install llama-cpp-python

# 또는 requirements 파일 사용
pip install -r requirements_sllm.txt
```

### 2. GPU 가속 (선택)

#### CUDA (NVIDIA GPU)
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

#### Metal (Mac M1/M2)
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

### 3. 모델 파일 배치

```
workspace_KcBERT/
└── models/
    └── Midm-2.0-Mini-Instruct-Q4_K_M.gguf  # 이미 있음
```

## 🚀 사용 방법

### 방법 1: Python 코드

```python
from src.detector_sllm import SLLMAbusiveDetector

# 초기화
detector = SLLMAbusiveDetector(
    model_path="./models/Midm-2.0-Mini-Instruct-Q4_K_M.gguf",
    threshold=0.5,
    n_threads=4  # CPU 스레드 수
)

# 텍스트 분석
result = detector.predict("분석할 통화 내용")

print(f"욕설 감지: {result['is_abusive']}")
print(f"점수: {result['abusive_score']:.4f}")
print(f"이유: {result['reason']}")
```

### 방법 2: 비교 테스트

KcBERT와 sLLM을 동시에 비교:

```bash
python compare_kcbert_sllm.py
```

## 📊 출력 예시

### 정상 통화
```
텍스트: "안녕하세요. 제품 문의 드립니다."

결과:
  판정: 정상
  점수: 0.1000
  이유: 정중한 문의 내용으로 욕설이나 공격적 표현 없음
  처리시간: 1.523초
```

### 욕설 포함 통화
```
텍스트: "야 거기 배송 왜 이렇게 느린거야? 이 병신들아."

결과:
  판정: 욕설
  점수: 0.9500
  이유: "병신" 등의 심각한 욕설 포함
  처리시간: 1.834초
```

## 🎨 프롬프트 엔지니어링

sLLM의 성능은 프롬프트에 크게 의존합니다. 현재 사용 중인 프롬프트:

```python
system_prompt = """당신은 고객 서비스 품질 관리 전문가입니다.
통화 내용을 분석하여 욕설, 폭언, 공격적인 언어가 있는지 판단해주세요.

평가 기준:
- 욕설/비속어 사용 (씨발, 병신, 개새끼 등)
- 위협적 표현 (죽이고 싶다, 때리고 싶다 등)
- 심한 모욕 (쓰레기, 인간말종 등)

정상적인 불만 표현은 욕설이 아닙니다:
- "답답하네요", "불편합니다", "개선 필요합니다" 등

반드시 다음 형식으로만 응답하세요:
점수: [0.0에서 1.0 사이의 숫자]
판단: [욕설 있음/욕설 없음]
이유: [간단한 이유]"""
```

### 프롬프트 커스터마이징

```python
detector = SLLMAbusiveDetector()
detector.system_prompt = """
당신만의 커스텀 프롬프트...
"""
```

## ⚙️ 파라미터 조정

### n_ctx (컨텍스트 길이)
```python
detector = SLLMAbusiveDetector(
    n_ctx=4096  # 기본: 2048
)
```
- 긴 통화 내용 처리 시 증가
- 메모리 사용량 증가

### temperature (온도)
```python
# detector_sllm.py 수정
response = self.llm(
    prompt,
    temperature=0.1  # 낮을수록 일관성 ↑, 창의성 ↓
)
```
- 0.0: 가장 결정적 (같은 입력 → 같은 출력)
- 1.0: 가장 창의적 (결과 가변적)
- **권장**: 0.1 (욕설 감지는 일관성 중요)

### n_threads (CPU 스레드)
```python
detector = SLLMAbusiveDetector(
    n_threads=8  # CPU 코어 수에 맞춰 조정
)
```

### n_gpu_layers (GPU 레이어)
```python
# detector_sllm.py 수정
self.llm = Llama(
    model_path=self.model_path,
    n_gpu_layers=35  # 0=CPU only, -1=전체 GPU
)
```

## 🔧 최적화 팁

### 1. CPU 최적화
```python
import os
os.environ['OMP_NUM_THREADS'] = '8'  # OpenMP 스레드
os.environ['MKL_NUM_THREADS'] = '8'  # Intel MKL

detector = SLLMAbusiveDetector(
    n_threads=8
)
```

### 2. GPU 가속
```bash
# CUDA 지원 버전 설치
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --force-reinstall llama-cpp-python
```

```python
# GPU 레이어 설정
detector = SLLMAbusiveDetector()
detector.llm = Llama(
    model_path="...",
    n_gpu_layers=-1  # 전체 GPU 사용
)
```

### 3. 배치 처리
```python
texts = [
    "통화 내용 1",
    "통화 내용 2",
    "통화 내용 3"
]

results = detector.predict_batch(texts)
```

## 📈 성능 벤치마크

테스트 환경: Intel i5-10400, 16GB RAM

| 설정 | 처리 시간/텍스트 | 메모리 |
|------|------------------|--------|
| CPU (4 threads) | 1.5-2.5초 | 3GB |
| CPU (8 threads) | 1.0-1.5초 | 3GB |
| GPU (CUDA) | 0.5-0.8초 | 4GB |

## 🎯 장단점

### ✅ 장점

1. **우수한 문맥 이해**
   - "미친듯이 좋다" vs "미친놈" 정확히 구분
   - 문장 전체의 의도 파악

2. **판단 근거 제공**
   - 왜 욕설로 판단했는지 이유 제공
   - 투명성과 신뢰성 향상

3. **Fine-tuning 불필요**
   - 사전 학습된 모델 바로 사용
   - 추가 학습 데이터 불필요

4. **온디바이스 실행**
   - 외부 API 불필요
   - 개인정보 보호

5. **유연한 기준 조정**
   - 프롬프트 수정만으로 기준 변경
   - 산업별 커스터마이징 쉬움

### ❌ 단점

1. **처리 속도**
   - KcBERT보다 5-10배 느림
   - 실시간 처리에는 부적합

2. **일관성**
   - 같은 입력에도 약간 다른 결과 가능
   - Temperature를 낮춰 완화 가능

3. **프롬프트 의존성**
   - 프롬프트 품질이 성능 결정
   - 지속적인 프롬프트 개선 필요

4. **모델 크기**
   - 2.5GB (KcBERT의 5배)
   - 배포 시 고려 필요

## 🔮 활용 시나리오

### KcBERT 사용이 적합한 경우

- ✅ 빠른 처리 속도 필요
- ✅ 대량 데이터 처리
- ✅ 실시간 감지
- ✅ 일관된 결과 필요

### sLLM 사용이 적합한 경우

- ✅ 정확도가 최우선
- ✅ 문맥 이해 중요
- ✅ 판단 근거 필요
- ✅ 오탐/미탐 최소화

### 하이브리드 접근

두 모델을 함께 사용:

```python
# 1차: KcBERT로 빠른 필터링
result_kcbert = detector_kcbert.predict(text)

# 2차: 경계 케이스만 sLLM으로 재검증
if 0.4 <= result_kcbert['abusive_score'] <= 0.6:
    result_sllm = detector_sllm.predict(text)
    final_result = result_sllm
else:
    final_result = result_kcbert
```

## 🐛 문제 해결

### 1. llama-cpp-python 설치 오류

```bash
# C++ 컴파일러 필요
# Windows: Visual Studio Build Tools 설치
# Linux: gcc, g++ 설치
# Mac: Xcode Command Line Tools

# 사전 빌드 버전 사용
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 2. 메모리 부족

```python
# 컨텍스트 길이 줄이기
detector = SLLMAbusiveDetector(
    n_ctx=1024  # 기본: 2048
)
```

### 3. 느린 속도

```python
# 스레드 수 증가
detector = SLLMAbusiveDetector(
    n_threads=os.cpu_count()
)

# GPU 사용 (CUDA 설치 후)
# detector_sllm.py에서 n_gpu_layers 조정
```

### 4. 응답 파싱 실패

프롬프트를 더 명확하게 수정하거나, `_parse_response` 메서드 개선

## 📚 참고 자료

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [GGUF 형식](https://github.com/ggerganov/llama.cpp/blob/master/docs/GGUF.md)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

**작성일**: 2026-01-21  
**버전**: 1.0  
**모델**: Midm-2.0-Mini-Instruct 4B (GGUF)
