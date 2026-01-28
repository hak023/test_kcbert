# KcBERT 욕설/폭언 감지 시스템

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![KcBERT](https://img.shields.io/badge/Model-KcBERT-orange.svg)

한국어 통화 내용에서 욕설 및 폭언을 자동으로 감지하는 AI 기반 시스템입니다.

## 🎯 주요 기능

- ✅ **KcBERT 기반 욕설 감지**: 한국어에 특화된 BERT 모델 활용
- ✅ **다중 카테고리 감지**: 욕설/폭언 + 성희롱 동시 판단 ⭐ NEW
- ✅ **sLLM 기반 감지**: 온디바이스 4B 모델로 문맥 이해 강화
- ✅ **Fine-tuning 비교**: 모델 학습 전후 성능 비교 ⭐ NEW
- ✅ **텍스트 파일 입력**: 통화 내용을 텍스트 파일로 입력
- ✅ **공격성 점수 산출**: 0~1 범위의 정량적 점수 제공
- ✅ **규칙 기반 보완**: 패턴 매칭으로 정확도 향상
- ✅ **배치 처리 지원**: 여러 파일 동시 분석 가능
- ✅ **결과 자동 저장**: JSON 형식으로 결과 저장
- ✅ **판단 근거 제공**: sLLM이 왜 그렇게 판단했는지 이유 설명

## 📋 시스템 요구사항

- **Python**: 3.8 이상
- **OS**: Windows / Linux / MacOS
- **RAM**: 최소 4GB, 권장 8GB
- **저장공간**: 약 1GB (모델 캐시 포함)
- **주의**: KcBERT 모델의 최대 입력 길이는 300 토큰입니다

## 🚀 빠른 시작

### 1. 설치

```powershell
# PowerShell에서 실행 (Windows)
.\install.ps1
```

또는 수동 설치:

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
.\venv\Scripts\Activate.ps1

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행

#### 방법 1: 대화형 실행 (권장)

```powershell
.\run.ps1
```

**실행 모드 선택**:
```
실행 모드를 선택하세요:
  1. 배치 처리 (모든 샘플 파일 자동 처리) ⭐ 권장
  2. 개별 파일 선택
  3. 다중 카테고리 테스트 (욕설 + 성희롱)
  4. Fine-tuning 전후 비교 테스트
  5. KcBERT vs sLLM 성능 비교
  6. 이슈 케이스 Fine-tuning 🔧
  7. Fine-tuned 모델 평가 📊
  8. 종료

선택 (1-8): 1
```

- **배치 처리 모드**: 모든 샘플을 한 번에 처리하고 요약 제공
- **개별 파일 모드**: 원하는 파일 하나만 분석
- **다중 카테고리 테스트**: 욕설과 성희롱을 동시에 감지하는 테스트
- **Fine-tuning 비교**: 모델 학습 전후 성능 비교
- **KcBERT vs sLLM 비교**: 20개 테스트 케이스로 두 모델 성능 비교
- **이슈 케이스 Fine-tuning**: 테스트 실패 케이스로 모델 재학습 🆕
- **Fine-tuned 모델 평가**: 재학습 전후 성능 비교 🆕

**배치 처리 결과 예시**:
```
📊 전체 처리 결과 요약
📁 처리된 파일: 4개
⚠️  욕설 감지: 1개
✅ 정상 통화: 3개
⏱️  총 처리 시간: 1.300초
📈 평균 처리 시간: 0.325초/파일
```

#### 방법 2: 배치 처리 직접 실행

모든 샘플 파일을 자동으로 순차 처리:

```powershell
python batch_process.py
```

#### 방법 3: 간편 실행 (개별 파일)

```powershell
# 예제 파일 분석
.\run_simple.ps1 data\samples\normal_call.txt

# 또는 직접 파일 경로 지정
.\run_simple.ps1 C:\my_data\call.txt
```

#### 방법 4: Python 직접 실행

```bash
# 기본 실행
python main.py --input data/samples/normal_call.txt

# 임계값 조정
python main.py --input data/samples/abusive_call.txt --threshold 0.7

# 결과 저장 위치 지정
python main.py --input test.txt --output results/my_result.json

# 결과 저장 안함
python main.py --input test.txt --no-save
```

## 📁 프로젝트 구조

```
workspace_KcBERT/
├── src/                          # 소스 코드
│   ├── __init__.py
│   ├── detector.py               # 메인 감지 엔진
│   ├── preprocessor.py           # 텍스트 전처리
│   ├── model_loader.py           # KcBERT 모델 로더
│   └── utils.py                  # 유틸리티 함수
├── data/                         # 데이터
│   ├── samples/                  # 예제 통화 내용
│   │   ├── normal_call.txt       # 정상 통화
│   │   ├── abusive_call.txt      # 욕설 포함 통화
│   │   ├── mixed_call.txt        # 혼합 통화
│   │   └── complaint_call.txt    # 불만 통화
│   └── results/                  # 분석 결과 저장
├── models/                       # 모델 캐시 (자동 생성)
├── docs/                         # 문서
│   └── design/
│       └── architecture.md       # 아키텍처 설계서
├── requirements.txt              # 의존성 패키지
├── config.yaml                   # 설정 파일
├── main.py                       # 메인 실행 스크립트
├── run.ps1                       # PowerShell 실행 스크립트 (대화형)
├── run_simple.ps1                # PowerShell 간편 실행 스크립트
├── install.ps1                   # 설치 스크립트
└── README.md                     # 본 문서
```

## 🎓 예제

### 예제 1: 정상 통화

**입력 파일**: `data/samples/normal_call.txt`

```
고객: 안녕하세요, 제품 문의 드립니다.
상담원: 네 안녕하세요. 무엇을 도와드릴까요?
고객: A 상품의 배송 기간이 궁금합니다.
상담원: A 상품은 주문 후 2-3일 이내에 배송됩니다.
```

**결과**:
```json
{
  "is_abusive": false,
  "abusive_score": 0.12,
  "confidence": 0.89
}
```

### 예제 2: 욕설 포함 통화

**입력 파일**: `data/samples/abusive_call.txt`

```
고객: 야 거기 배송 왜 이렇게 느린거야?
고객: 이 병신들아. 빨리 안되냐고.
고객: 시끄러워. 너네 개같은 서비스 때문에...
```

**결과**:
```json
{
  "is_abusive": true,
  "abusive_score": 0.87,
  "confidence": 0.94
}
```

## ⚙️ 설정

`config.yaml` 파일에서 시스템 동작을 커스터마이즈할 수 있습니다:

```yaml
model:
  name: "beomi/kcbert-base"  # 사용할 모델
  max_length: 512            # 최대 토큰 길이

detection:
  threshold: 0.5             # 감지 임계값 (0.0 ~ 1.0)
  
output:
  save_results: true         # 결과 자동 저장
  results_dir: "./data/results"
```

### 임계값 조정 가이드

- **0.3**: 매우 민감 (경미한 불만도 감지)
- **0.5**: 균형 (기본값, 권장)
- **0.7**: 보수적 (명확한 욕설만 감지)

## 🔧 고급 사용법

### API 방식 사용

Python 코드에서 직접 사용:

```python
# 기본 욕설 감지
from src.detector import AbusiveDetector

detector = AbusiveDetector(threshold=0.5)
result = detector.predict("분석할 텍스트")

print(f"욕설 감지: {result['is_abusive']}")
print(f"공격성 점수: {result['abusive_score']}")
```

```python
# 다중 카테고리 감지 (욕설 + 성희롱) ⭐ NEW
from src.detector_multi import MultiCategoryDetector

detector = MultiCategoryDetector()
result = detector.predict("분석할 텍스트")

print(f"욕설: {result['is_abusive']} ({result['abusive_score']:.2f})")
print(f"성희롱: {result['is_sexual_harassment']} ({result['harassment_score']:.2f})")
print(f"카테고리: {result['categories']}")
```

### 배치 처리

```python
texts = [
    "첫 번째 통화 내용",
    "두 번째 통화 내용",
    "세 번째 통화 내용"
]

results = detector.predict_batch(texts)
```

## 📊 성능

- **처리 속도**: 텍스트당 약 0.5~1초 (CPU 기준)
- **정확도**: 규칙 기반과 결합하여 약 85% (Fine-tuning 시 향상 가능)
- **메모리**: 약 1.5GB (모델 로드 시)

## ⚠️ 주의사항

1. **첫 실행 시 시간 소요**: 
   - 설치 후 첫 실행 시 PyTorch/Transformers 모듈 로딩에 약 40초~1분 소요
   - 두 번째 실행부터는 빠르게 동작합니다

2. **Fine-tuning 권장**: 현재 버전은 기본 KcBERT를 사용합니다. 실제 운영 환경에서는 실제 통화 데이터로 Fine-tuning한 모델을 사용하는 것을 권장합니다.

3. **문맥 의존성**: 문맥에 따라 오탐/미탐이 발생할 수 있습니다. 규칙 기반 필터링으로 일부 보완하고 있습니다.

4. **텍스트 길이 제한**: KcBERT는 최대 300 토큰까지 처리 가능합니다 (약 200-250 단어).

5. **개인정보 보호**: 통화 내용에 개인정보가 포함되어 있다면 적절히 마스킹 후 사용하세요.

## 🔮 향후 계획

- [ ] 실제 통화 데이터로 Fine-tuning
- [ ] 웹 API 서버 구축
- [ ] 실시간 스트리밍 분석 지원
- [ ] 감정 분석 기능 추가
- [ ] 다국어 지원 (영어, 일본어 등)
- [ ] 대시보드 UI 개발

## 📖 참고 자료

- **KcBERT GitHub**: https://github.com/Beomi/KcBERT
- **Hugging Face Model**: https://huggingface.co/beomi/kcbert-base
- **설계 문서**: `docs/design/architecture.md`
- **사용 가이드**: `docs/guides/usage.md`
- **sLLM 가이드**: `docs/guides/sllm_detector.md`
- **성희롱 감지**: `docs/guides/sexual_harassment_detection.md` ⭐
- **Fine-tuning 가이드**: `docs/guides/fine_tuning_explained.md` ⭐
- **Fine-tuning 비교**: `docs/guides/finetuning_comparison_test.md` ⭐
- **이슈 케이스 Fine-tuning**: `docs/guides/issue_cases_finetuning.md` 🆕
- **sLLM 프롬프트 개선**: `docs/SLLM_PROMPT_IMPROVED.md` ⭐
- **모델 성능 비교**: `docs/guides/model_comparison.md` 🆕
- **성능 분석 보고서**: `docs/PERFORMANCE_ANALYSIS_REPORT.md` 🆕
- **Fine-tuned KcBERT vs sLLM 비교 리포트**: `docs/KCBERT_FINETUNED_VS_SLLM_REPORT.md` 🆕✨
- **최종 비교 리포트**: `docs/FINAL_COMPARISON_REPORT.md` 📊
- **성능 최적화**: `docs/guides/performance_optimization.md`
- **정확도 개선**: `docs/guides/accuracy_improvement.md`

## 🤝 기여

버그 리포트, 기능 제안, Pull Request 등 모든 기여를 환영합니다!

## 📝 라이선스

MIT License

## 👥 문의

문제가 발생하거나 질문이 있으시면 Issue를 등록해주세요.

---

**Made with ❤️ using KcBERT + sLLM**
