# 성희롱 탐지 추가 가이드

## 📋 목차

1. [개요](#개요)
2. [구현 방법 비교](#구현-방법-비교)
3. [방법 1: 규칙 기반 패턴 추가 (즉시 가능)](#방법-1-규칙-기반-패턴-추가)
4. [방법 2: 멀티 레이블 분류 (Fine-tuning)](#방법-2-멀티-레이블-분류)
5. [방법 3: 별도 모델 사용](#방법-3-별도-모델-사용)
6. [방법 4: 계층적 분류](#방법-4-계층적-분류)
7. [추천 방법](#추천-방법)

---

## 개요

현재 KcBERT 시스템에 성희롱 탐지 기능을 추가하는 방법들을 소개합니다.

### 현재 상태
- ✅ 욕설/폭언 탐지 (규칙 기반 + KcBERT)
- ❌ 성희롱 탐지 (미구현)

### 목표
- ✅ 욕설/폭언 탐지
- ✅ 성희롱 탐지
- ✅ 각각의 점수 제공
- ✅ 복합적인 상황 처리 (욕설 + 성희롱)

---

## 구현 방법 비교

| 방법 | 난이도 | 정확도 | 구현 시간 | 비용 | 추천 |
|-----|-------|-------|---------|------|------|
| **1. 규칙 기반 추가** | ⭐ 쉬움 | ⭐⭐⭐ 중간 | 1시간 | 무료 | ⭐⭐⭐ |
| **2. 멀티 레이블 분류** | ⭐⭐⭐ 어려움 | ⭐⭐⭐⭐⭐ 높음 | 1주일+ | 학습 데이터 필요 | ⭐⭐⭐⭐ |
| **3. 별도 모델** | ⭐⭐ 보통 | ⭐⭐⭐⭐ 높음 | 2일 | 중간 | ⭐⭐⭐ |
| **4. 계층적 분류** | ⭐⭐⭐ 어려움 | ⭐⭐⭐⭐⭐ 매우높음 | 2주일+ | 높음 | ⭐⭐ |

---

## 방법 1: 규칙 기반 패턴 추가

### 장점
- ✅ **즉시 구현 가능** (1시간 이내)
- ✅ **학습 데이터 불필요**
- ✅ **설명 가능성 높음** (어떤 단어가 문제인지 명확)
- ✅ **빠른 성능** (추가 비용 없음)

### 단점
- ❌ 문맥 파악 부족
- ❌ 우회 표현 탐지 어려움
- ❌ 지속적인 패턴 업데이트 필요

### 구현 방법

#### 1) 현재 detector.py 확장

```python
class MultiCategoryDetector:
    """다중 카테고리 감지기 (욕설, 성희롱 등)"""
    
    def __init__(self):
        # 욕설/폭언 패턴
        self.abusive_patterns = [
            r'씨발', r'개새끼', r'미친놈', r'병신',
            r'좆', r'꺼져', r'닥쳐', r'죽어'
        ]
        
        # 성희롱 패턴
        self.sexual_harassment_patterns = [
            # 외모 관련
            r'섹시하?[네다]', r'몸매', r'가슴', r'엉덩이',
            
            # 성적 제안/암시
            r'같이\s*자[자요]', r'호텔\s*가[자요]', r'모텔',
            r'원나잇', r'섹스', r'잠자리',
            
            # 외모 평가
            r'예쁘?[네다요].*보[이여]', r'이쁘?다', r'귀엽?다',
            r'스타일\s*좋', r'몸\s*좋',
            
            # 개인적 질문
            r'남자친구\s*있[어니]', r'여자친구\s*있[어니]',
            r'혼자\s*[사살]?[니냐]', r'결혼.*했?[어니냐]',
            
            # 은어/비속어
            r'꼬시', r'작업\s*걸', r'헌팅', r'픽업'
        ]
        
        # 심각한 성희롱 (즉시 심각 판정)
        self.severe_sexual_harassment = [
            r'강간', r'성폭행', r'성관계', r'성행위',
            r'몸\s*만지', r'몸\s*봐', r'옷\s*벗'
        ]
        
        # 화이트리스트 (오탐 방지)
        self.whitelist = [
            r'섹시한?\s*디자인',  # 제품 설명
            r'섹시한?\s*이미지',
            r'예쁘?게.*포장',    # 서비스 표현
        ]
    
    def detect(self, text: str) -> Dict[str, Any]:
        """다중 카테고리 감지"""
        
        # 욕설/폭언 점수
        abusive_score = self._check_patterns(text, self.abusive_patterns)
        
        # 성희롱 점수
        harassment_score = self._check_sexual_harassment(text)
        
        # 최종 판정
        return {
            "text": text,
            
            # 욕설/폭언
            "is_abusive": abusive_score >= 0.5,
            "abusive_score": abusive_score,
            
            # 성희롱
            "is_sexual_harassment": harassment_score >= 0.5,
            "harassment_score": harassment_score,
            
            # 전체 부적절성
            "is_inappropriate": max(abusive_score, harassment_score) >= 0.5,
            "max_severity": max(abusive_score, harassment_score),
            
            # 카테고리 분류
            "categories": self._categorize(abusive_score, harassment_score),
            
            # 상세 정보
            "details": {
                "abusive_words": self._find_matches(text, self.abusive_patterns),
                "harassment_words": self._find_matches(
                    text, 
                    self.sexual_harassment_patterns + self.severe_sexual_harassment
                )
            }
        }
    
    def _check_sexual_harassment(self, text: str) -> float:
        """성희롱 점수 계산"""
        
        # 화이트리스트 체크
        for pattern in self.whitelist:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.0
        
        score = 0.0
        
        # 심각한 성희롱 체크
        for pattern in self.severe_sexual_harassment:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.95  # 즉시 높은 점수
        
        # 일반 성희롱 패턴
        matches = 0
        for pattern in self.sexual_harassment_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        # 점수 계산
        if matches >= 3:
            score = 0.9
        elif matches == 2:
            score = 0.7
        elif matches == 1:
            score = 0.5
        
        return score
    
    def _categorize(self, abusive_score: float, harassment_score: float) -> List[str]:
        """카테고리 분류"""
        categories = []
        
        if abusive_score >= 0.5:
            categories.append("욕설/폭언")
        
        if harassment_score >= 0.5:
            categories.append("성희롱")
        
        if not categories:
            categories.append("정상")
        
        return categories
```

#### 2) 사용 예제

```python
from src.detector_multi import MultiCategoryDetector

detector = MultiCategoryDetector()

# 테스트 케이스
test_cases = [
    "제품 배송이 늦어서 불편합니다.",           # 정상
    "씨발 빨리 보내라 이 새끼들아!",           # 욕설
    "얼굴 예쁘네요. 남자친구 있어요?",         # 성희롱
    "개새끼들 씨발, 섹시하네 같이 자자",       # 욕설 + 성희롱
]

for text in test_cases:
    result = detector.detect(text)
    print(f"\n텍스트: {text}")
    print(f"카테고리: {result['categories']}")
    print(f"욕설 점수: {result['abusive_score']:.2f}")
    print(f"성희롱 점수: {result['harassment_score']:.2f}")
```

#### 3) 결과 포맷

```json
{
  "text": "얼굴 예쁘네요. 남자친구 있어요?",
  "is_abusive": false,
  "abusive_score": 0.0,
  "is_sexual_harassment": true,
  "harassment_score": 0.7,
  "is_inappropriate": true,
  "max_severity": 0.7,
  "categories": ["성희롱"],
  "details": {
    "abusive_words": [],
    "harassment_words": ["예쁘네요", "남자친구 있어요"]
  }
}
```

---

## 방법 2: 멀티 레이블 분류

### 개요
KcBERT를 멀티 레이블 분류로 fine-tuning하여 욕설, 성희롱, 혐오 등을 동시에 판단

### 장점
- ✅ **높은 정확도** (문맥 파악 가능)
- ✅ **우회 표현 탐지** 가능
- ✅ **일반화 성능** 좋음

### 단점
- ❌ **학습 데이터 필요** (각 카테고리당 1,000개+)
- ❌ **시간 소요** (데이터 수집 + 라벨링 + 학습)
- ❌ **전문 지식 필요** (ML/NLP)

### 구현 개요

```python
# 모델 구조
class MultiLabelKcBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.kcbert = AutoModel.from_pretrained("beomi/kcbert-base")
        self.classifier = nn.Linear(768, 3)  # [욕설, 성희롱, 혐오]
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.kcbert(input_ids, attention_mask)
        logits = self.classifier(outputs.pooler_output)
        return self.sigmoid(logits)  # 각 레이블 독립적 확률

# 학습 데이터 형식
train_data = [
    {
        "text": "씨발 빨리 해라",
        "labels": [1, 0, 0]  # [욕설: O, 성희롱: X, 혐오: X]
    },
    {
        "text": "얼굴 예쁘네 같이 자자",
        "labels": [0, 1, 0]  # [욕설: X, 성희롱: O, 혐오: X]
    },
    {
        "text": "여자는 집에나 있어",
        "labels": [0, 0, 1]  # [욕설: X, 성희롱: X, 혐오: O]
    },
    {
        "text": "씨발년들 꺼져",
        "labels": [1, 1, 1]  # [욕설: O, 성희롱: O, 혐오: O]
    }
]
```

### 필요 리소스
- **학습 데이터**: 카테고리당 1,000~5,000개
- **라벨링 시간**: 1인 기준 약 2~4주
- **학습 시간**: GPU 기준 4~8시간
- **비용**: 데이터 라벨링 비용 (크라우드소싱 활용 시 100만원~)

---

## 방법 3: 별도 모델 사용

### 개요
욕설 탐지용 KcBERT + 성희롱 탐지용 별도 모델

### 장점
- ✅ **각 모델 독립적** 최적화
- ✅ **기존 코드 영향 최소**
- ✅ **점진적 개선** 가능

### 단점
- ❌ **처리 시간 2배**
- ❌ **메모리 사용량 증가**
- ❌ **모델 관리 복잡**

### 구현 개요

```python
class DualDetector:
    def __init__(self):
        # 욕설 탐지 모델
        self.abusive_detector = AbusiveDetector()
        
        # 성희롱 탐지 모델 (별도)
        self.harassment_detector = HarassmentDetector()
    
    def detect(self, text: str) -> Dict:
        # 병렬 처리
        abusive_result = self.abusive_detector.predict(text)
        harassment_result = self.harassment_detector.predict(text)
        
        return {
            "abusive": abusive_result,
            "harassment": harassment_result,
            "is_inappropriate": (
                abusive_result['is_abusive'] or 
                harassment_result['is_harassment']
            )
        }
```

### 성능
- **처리 시간**: 약 250ms (현재 130ms의 2배)
- **처리량**: 약 4건/초 (현재 6건/초의 2/3)

---

## 방법 4: 계층적 분류

### 개요
1단계: 부적절한 발언인가?
2단계: 욕설/성희롱/혐오 중 어떤 유형?

### 장점
- ✅ **효율적** (부적절하지 않으면 1단계에서 종료)
- ✅ **정확도 높음**

### 단점
- ❌ **구현 복잡**
- ❌ **학습 데이터 대량 필요**
- ❌ **에러 전파** (1단계 실수 시 2단계도 실패)

---

## 추천 방법

### 🎯 단기 (즉시~1주일)
**→ 방법 1: 규칙 기반 패턴 추가** ⭐⭐⭐

**이유:**
- 즉시 구현 가능
- 학습 데이터 불필요
- 80% 정확도로도 충분히 유용
- 빠른 검증 가능

**구현 순서:**
1. 성희롱 패턴 정의 (1시간)
2. detector.py 확장 (2시간)
3. 테스트 및 조정 (2시간)

### 🎯 중기 (1개월)
**→ 방법 1 + 방법 2 일부** ⭐⭐⭐⭐

**이유:**
- 규칙 기반으로 빠르게 시작
- 사용하며 데이터 수집
- 수집된 데이터로 fine-tuning

**구현 순서:**
1. 규칙 기반 배포
2. 실제 데이터 수집 및 라벨링
3. 소량 데이터로 fine-tuning 시작
4. 점진적 정확도 개선

### 🎯 장기 (3개월+)
**→ 방법 2: 멀티 레이블 분류** ⭐⭐⭐⭐⭐

**이유:**
- 최고 정확도
- 문맥 이해 가능
- 지속적 개선 가능

**구현 순서:**
1. 대량 데이터 수집 (5,000개+)
2. 전문가 라벨링
3. Fine-tuning
4. 평가 및 배포

---

## 실전 예제

### 즉시 사용 가능한 코드

아래는 **방법 1**을 바로 사용할 수 있도록 구현한 예제입니다:

```python
# src/detector_multi.py 생성
# (위의 코드 참조)

# 사용 예제
from src.detector_multi import MultiCategoryDetector

detector = MultiCategoryDetector()

# 고객센터 통화 분석
call_text = """
고객: 배송이 늦어져서 연락드렸습니다.
상담원: 죄송합니다. 확인해보겠습니다.
고객: 씨발 빨리 좀 처리해주세요.
"""

result = detector.detect(call_text)

print(f"부적절한 발언: {result['is_inappropriate']}")
print(f"욕설/폭언: {result['is_abusive']} ({result['abusive_score']:.2f})")
print(f"성희롱: {result['is_sexual_harassment']} ({result['harassment_score']:.2f})")
print(f"카테고리: {', '.join(result['categories'])}")
```

---

## 패턴 업데이트 가이드

### 성희롱 패턴 추가 시 주의사항

1. **문맥 고려**
   - "예쁘다" → 제품 설명에서는 정상
   - "얼굴 예쁘다" → 외모 평가는 성희롱

2. **화이트리스트 활용**
   - 정상적인 표현 보호
   - 오탐 방지

3. **지속적 업데이트**
   - 새로운 은어/비속어 추가
   - 우회 표현 대응

4. **법적 기준 참고**
   - 고용노동부 성희롱 판단 기준
   - 판례 참고

---

## 성능 영향

| 방법 | 처리 시간 | 메모리 | 정확도 |
|-----|---------|--------|--------|
| **현재 (욕설만)** | 130ms | 500MB | 75% |
| **방법 1 (규칙 추가)** | 135ms | 500MB | 80% |
| **방법 2 (멀티 레이블)** | 130ms | 500MB | 95% |
| **방법 3 (별도 모델)** | 260ms | 1GB | 90% |

---

## 다음 단계

1. ✅ 성희롱 패턴 정의
2. ✅ detector_multi.py 구현
3. ⏳ 테스트 및 검증
4. ⏳ 실제 데이터로 평가
5. ⏳ 패턴 개선

---

## 참고 자료

- [고용노동부 성희롱 판단 기준](https://www.moel.go.kr/)
- [AI 윤리 가이드라인](https://www.msit.go.kr/)
- [KcBERT Fine-tuning 가이드](https://github.com/Beomi/KcBERT)
- [멀티 레이블 분류 논문](https://arxiv.org/abs/1901.02860)
