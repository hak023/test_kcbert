# KcBERT vs sLLM 성능 비교 가이드

## 📅 작성일
2026-01-27

## 📋 개요

KcBERT 모델과 sLLM 모델의 성능을 20개의 다양한 테스트 케이스로 비교하여, 각 모델의 강점과 약점을 파악합니다.

---

## 🎯 비교 목적

### 1. 처리 속도 비교
- KcBERT: 경량 BERT 모델, CPU 최적화
- sLLM: 4B 파라미터 언어 모델, 컨텍스트 이해 우수

### 2. 정확도 비교
- 다양한 부적절 표현 감지 능력
- 경계선 케이스 판단 능력
- 오탐률 및 미탐률 비교

### 3. 실용성 평가
- 실시간 처리 vs 배치 처리
- 리소스 사용량
- 적용 시나리오 추천

---

## 📊 테스트 케이스 (20개)

### 정상 케이스 (4개)
```
test_01_normal_service.txt       - 정상적인 서비스 문의
test_14_polite_complaint.txt     - 정중한 불만
test_15_urgent_request.txt       - 긴급 요청
test_20_appreciation.txt         - 감사 표현
```

### 경계선 케이스 (5개)
```
test_02_strong_complaint.txt     - 강한 불만 (욕설 없음)
test_09_borderline_angry.txt     - 화난 표현 (경계)
test_10_borderline_frustrated.txt- 답답한 표현 (경계)
test_11_threat_legal.txt         - 법적 조치 언급
test_19_emotional_outburst.txt   - 감정적 폭발
```

### 명확한 부적절 케이스 (11개)

#### 욕설/폭언 (3개)
```
test_03_explicit_profanity.txt   - 명시적 욕설
test_04_insult_no_swear.txt      - 모욕 (욕설 없음)
test_17_mild_insult.txt          - 경미한 모욕
```

#### 성희롱 (2개)
```
test_06_sexual_harassment_direct.txt  - 직접적 성희롱
test_07_sexual_harassment_subtle.txt  - 우회적 성희롱
```

#### 위협 (2개)
```
test_05_direct_threat.txt        - 직접적 위협
test_18_explicit_threat.txt      - 명시적 위협
```

#### 복합/특수 (4개)
```
test_08_sarcastic_insult.txt     - 비꼬는 모욕
test_12_mixed_profanity_threat.txt - 욕설 + 위협
test_13_profanity_sexual.txt     - 욕설 + 성희롱
test_16_passive_aggressive.txt   - 수동공격적 표현
```

---

## 🚀 실행 방법

### 방법 1: PowerShell 스크립트 사용 (권장)

```powershell
# run.ps1 실행
.\run.ps1

# 메뉴에서 선택
5. KcBERT vs sLLM 성능 비교 🆕
```

### 방법 2: 직접 실행

```powershell
# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 비교 스크립트 실행
python compare_kcbert_vs_sllm.py
```

---

## ⏱️ 예상 소요 시간

```
KcBERT: 약 10-20초 (20개 파일)
sLLM:   약 10-20분 (20개 파일)
총:     약 10-20분

* sLLM은 CPU에서 실행 시 파일당 30초~1분 소요
```

---

## 📈 출력 결과

### 1. 실시간 진행 상황

```
[1/20] test_01_normal_service.txt
────────────────────────────────────────────────────────────────────────────────
📝 내용: "안녕하세요. 배송 문의 드리려고 전화했습니다..."

  🔵 KcBERT 분석 중... 완료 (0.45초)
     점수: 0.123
     판정: 정상

  🟢 sLLM 분석 중... 완료 (35.67초)
     점수: 0.150
     판정: 정상
     카테고리: 없음
```

### 2. 통계 요약

```
📊 전체 통계
────────────────────────────────────────────────────────────────────────────────

  ⏱️  처리 시간 비교
     KcBERT: 10.23초 (평균 0.51초/건)
     sLLM:   1,245.67초 (평균 62.28초/건)
     배속:   sLLM이 KcBERT보다 122.0x 느림

  🎯 정확도 비교
     KcBERT: 75.0%
     sLLM:   85.0%
     차이:   10.0%p (sLLM 우수)

  📏 점수 오차 (MAE)
     KcBERT: 0.125
     sLLM:   0.087
     차이:   0.038 (sLLM 우수)

  📂 sLLM 카테고리별 감지
     ├─ 없음: 4건
     ├─ 불만: 5건
     ├─ 욕설: 3건
     ├─ 성희롱: 2건
     ├─ 위협: 2건
     ├─ 모욕: 3건
     └─ 복합: 1건
```

### 3. 상세 비교표

```
파일명                                | 실제     | KcBERT   | sLLM     | 카테고리
────────────────────────────────────────────────────────────────────────────────
test_01_normal_service.txt            |   0.10  |  0.123  |  0.150  |   없음
test_02_strong_complaint.txt          |   0.40  |  0.456  |  0.420  |   불만
test_03_explicit_profanity.txt        |   0.95  |  0.892  |  0.950  |   욕설
test_04_insult_no_swear.txt           |   0.85  |  0.654  |  0.820  |   모욕
test_05_direct_threat.txt             |   0.90  |  0.723  |  0.880  |   위협
...
```

### 4. 최종 결론

```
🏆 종합 평가
────────────────────────────────────────────────────────────────────────────────

  ⚡ 속도: KcBERT 승 (122.0x 빠름)
  🎯 정확도: sLLM 승 (10.0%p 우수)
  📏 점수 정확성: sLLM 승

  💡 권장 사항
────────────────────────────────────────────────────────────────────────────────
  ✅ 대량 처리: KcBERT (속도 우수)
  ✅ 정밀 분석: sLLM (카테고리 분류 가능)
```

---

## 📊 평가 지표 설명

### 1. 정확도 (Accuracy)
```
정의: 올바르게 분류한 케이스 비율

계산: (정답 수 / 전체 수) × 100

분류 기준:
- 점수 < 0.3: 정상
- 0.3 ≤ 점수 < 0.6: 경계선
- 점수 ≥ 0.6: 부적절
```

### 2. MAE (Mean Absolute Error)
```
정의: 예측 점수와 실제 점수의 평균 절대 오차

계산: Σ|예측 - 실제| / 케이스 수

해석:
- 0.0 ~ 0.1: 우수
- 0.1 ~ 0.2: 양호
- 0.2 ~ 0.3: 보통
- 0.3 이상: 개선 필요
```

### 3. 처리 시간
```
측정: 각 파일당 분석 소요 시간

지표:
- 총 시간: 전체 파일 처리 시간
- 평균 시간: 파일당 평균 시간
- 배속: 두 모델 간 속도 비율
```

---

## 🔍 예상 결과 분석

### KcBERT 강점
```
✅ 속도: 100배 이상 빠름
✅ 명시적 욕설: 90% 이상 감지
✅ 리소스: CPU 최적화
✅ 안정성: 일관된 성능

약점:
❌ 문맥 이해: 제한적
❌ 우회 표현: 감지 어려움
❌ 성희롱: 감지율 낮음 (30-40%)
❌ 비꼬기: 감지 어려움
```

### sLLM 강점
```
✅ 정확도: 10-15% 높음
✅ 문맥 이해: 우수
✅ 우회 표현: 감지 가능
✅ 카테고리: 자동 분류
✅ 설명: 판단 근거 제공

약점:
❌ 속도: 매우 느림 (100배)
❌ 리소스: 메모리 많이 사용
❌ 일관성: 약간 변동 가능
```

---

## 💡 사용 시나리오별 추천

### 시나리오 1: 실시간 모니터링
```
추천: KcBERT
이유:
- 빠른 응답 시간 필요
- 대량 데이터 처리
- 명시적 욕설 감지로 충분
```

### 시나리오 2: 정밀 분석
```
추천: sLLM
이유:
- 높은 정확도 필요
- 카테고리 분류 필요
- 우회적 표현 감지 필요
- 시간 여유 있음
```

### 시나리오 3: 하이브리드 접근
```
추천: KcBERT (1차) + sLLM (2차)

프로세스:
1. KcBERT로 전체 스크리닝
2. 의심 케이스 (점수 0.4-0.7)만 sLLM 재검증
3. 명확한 케이스는 KcBERT 결과 사용

장점:
✅ 속도와 정확도 균형
✅ 비용 효율적
✅ 높은 신뢰도
```

### 시나리오 4: 대량 배치 처리
```
추천: KcBERT
이유:
- 수천~수만 건 처리
- 실시간 아닌 배치 작업
- sLLM으로는 시간 소요 과다
```

---

## 📁 결과 파일

### 저장 위치
```
data/results/comparison_kcbert_vs_sllm_YYYYMMDD_HHMMSS.json
```

### 파일 구조
```json
{
  "timestamp": "20260127_143052",
  "test_count": 20,
  "summary": {
    "kcbert": {
      "total_time": 10.23,
      "avg_time": 0.51,
      "accuracy": 75.0,
      "mae": 0.125
    },
    "sllm": {
      "total_time": 1245.67,
      "avg_time": 62.28,
      "accuracy": 85.0,
      "mae": 0.087,
      "categories": {
        "없음": 4,
        "불만": 5,
        "욕설": 3,
        ...
      }
    }
  },
  "kcbert_results": { ... },
  "sllm_results": { ... },
  "ground_truth": { ... }
}
```

---

## 🔧 고급 활용

### 1. 임계값 조정
```python
# compare_kcbert_vs_sllm.py 수정
GROUND_TRUTH = {
    "test_01_normal_service.txt": {
        "label": "정상",
        "score": 0.1,  # 조정 가능
        "category": "없음"
    },
    ...
}
```

### 2. 커스텀 테스트 케이스 추가
```bash
# data/samples/ 폴더에 test_*.txt 파일 추가
# 예: test_21_custom_case.txt

# GROUND_TRUTH에 추가
"test_21_custom_case.txt": {
    "label": "부적절",
    "score": 0.80,
    "category": "욕설"
}
```

### 3. 특정 케이스만 테스트
```python
# compare_kcbert_vs_sllm.py 수정
test_files = [
    "test_03_explicit_profanity.txt",
    "test_06_sexual_harassment_direct.txt",
    # 원하는 파일만 추가
]
```

---

## ⚠️ 주의사항

### 1. sLLM 모델 필요
```bash
# sLLM 모델 파일 확인
models/Midm-2.0-Mini-Instruct-Q4_K_M.gguf

# llama-cpp-python 설치 확인
pip install llama-cpp-python
```

### 2. 메모리 요구사항
```
KcBERT: ~2GB RAM
sLLM:   ~4GB RAM
총:     최소 6GB RAM 권장
```

### 3. 처리 시간
```
20개 파일 기준:
- KcBERT: 10-20초
- sLLM:   10-20분
- 총:     최소 10분 소요 예상
```

---

## 📚 관련 문서

- `docs/guides/sllm_detector.md` - sLLM 사용 가이드
- `docs/SLLM_PROMPT_IMPROVED.md` - 개선된 프롬프트
- `docs/guides/sexual_harassment_detection.md` - 성희롱 감지
- `README.md` - 프로젝트 전체 개요

---

## 🎯 결론

### KcBERT
```
✅ 실시간 처리, 대량 데이터에 적합
✅ 명시적 욕설 감지 우수
✅ 빠르고 안정적
```

### sLLM
```
✅ 정밀 분석, 복잡한 케이스에 적합
✅ 문맥 이해 및 카테고리 분류 우수
✅ 우회 표현 감지 가능
```

### 권장
```
💡 상황에 따라 선택 또는 하이브리드 접근 사용
💡 실시간: KcBERT
💡 정밀: sLLM
💡 균형: KcBERT (1차) + sLLM (2차)
```

---

**작성일**: 2026-01-27  
**작성자**: AI Assistant  
**버전**: 1.0  
**상태**: ✅ 완료
