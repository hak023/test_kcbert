# -*- coding: utf-8 -*-
"""
개선된 sLLM 프롬프트 테스트
문맥과 의도를 파악하는 능력 검증
"""

import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(case, result):
    """결과 출력"""
    print(f"📝 텍스트: \"{case['text']}\"")
    print(f"📊 예상: {case['expected']}")
    print()
    
    # 결과
    if result['is_abusive']:
        print(f"  🔴 판정: 부적절 (점수: {result['abusive_score']:.2f})")
    else:
        print(f"  ✅ 판정: 정상 (점수: {result['abusive_score']:.2f})")
    
    print(f"  📂 카테고리: {result.get('category', 'N/A')}")
    print(f"  💭 이유: {result['reason']}")
    print(f"  ⏱️  처리 시간: {result['processing_time']:.2f}초")
    
    # 정확도 판단
    is_correct = (
        (case['expected'] == '부적절' and result['is_abusive']) or
        (case['expected'] == '정상' and not result['is_abusive'])
    )
    
    if is_correct:
        print("  ✅ 정확!")
    else:
        print("  ❌ 부정확!")
    
    print()
    print(f"  📄 원본 응답:")
    for line in result['raw_response'].split('\n'):
        if line.strip():
            print(f"     {line}")
    print()


def main():
    """메인 함수"""
    print_header("🔬 개선된 sLLM 프롬프트 테스트")
    
    print("이 테스트는 개선된 프롬프트가 문맥과 의도를 파악하는지 검증합니다.")
    print()
    
    # 테스트 케이스
    test_cases = [
        # 명확한 정상
        {
            "text": "배송이 늦어서 불편합니다. 빠른 처리 부탁드립니다.",
            "expected": "정상",
            "description": "정상적인 불만 표현"
        },
        {
            "text": "상담사님 친절하시네요. 감사합니다.",
            "expected": "정상",
            "description": "긍정적인 표현"
        },
        
        # 명확한 욕설
        {
            "text": "씨발 빨리 좀 해라. 개새끼들아.",
            "expected": "부적절",
            "description": "명시적 욕설"
        },
        
        # 모욕적 표현
        {
            "text": "진짜 무능한 것들. 일 제대로 하는 게 없네. 쓰레기 같은 회사.",
            "expected": "부적절",
            "description": "모욕적 표현 (욕설 없음)"
        },
        
        # 성희롱 - 직접적
        {
            "text": "목소리 좋으시네요. 얼굴도 예쁘시겠어요. 남자친구 있어요?",
            "expected": "부적절",
            "description": "직접적 성희롱"
        },
        
        # 성희롱 - 우회적
        {
            "text": "와 진짜 목소리 너무 좋으신데요. 혹시 퇴근하고 시간 되세요? 저녁이나 같이 하면 안 될까요?",
            "expected": "부적절",
            "description": "우회적 성희롱 (제안)"
        },
        
        # 위협적 표현
        {
            "text": "가만 안 둔다. 이번주 안에 해결 안 되면 본사 찾아갈 거야.",
            "expected": "부적절",
            "description": "위협적 표현"
        },
        
        # 경계선 케이스 1 - 강한 불만
        {
            "text": "진짜 답답하네요. 이게 말이 됩니까? 너무 실망스럽습니다.",
            "expected": "정상",
            "description": "강한 불만이지만 정상 표현"
        },
        
        # 경계선 케이스 2 - 비꼬는 표현
        {
            "text": "와 정말 대단하시네요. 이렇게 일 못하는 회사는 처음 봅니다. 정말 훌륭하십니다.",
            "expected": "부적절",
            "description": "비꼬는 표현 (의도 파악 필요)"
        },
        
        # 복합 - 욕설 + 성희롱
        {
            "text": "씨발 이 예쁜 것아. 몸매 좋은데 같이 한잔 하자.",
            "expected": "부적절",
            "description": "욕설 + 성희롱 복합"
        }
    ]
    
    print(f"📝 테스트 케이스: {len(test_cases)}개")
    print()
    
    # 모델 로딩
    print("📥 sLLM 모델 로딩 중...")
    from src.detector_sllm import SLLMAbusiveDetector
    
    detector = SLLMAbusiveDetector(verbose=False)
    
    print("\n" + "=" * 70)
    print()
    
    # 테스트 실행
    results = []
    correct_count = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['description']}")
        print("-" * 70)
        
        result = detector.predict(case['text'])
        
        # 정확도 판단
        is_correct = (
            (case['expected'] == '부적절' and result['is_abusive']) or
            (case['expected'] == '정상' and not result['is_abusive'])
        )
        
        if is_correct:
            correct_count += 1
        
        results.append({
            'case': case,
            'result': result,
            'correct': is_correct
        })
        
        print_result(case, result)
        print()
    
    # 통계
    print_header("📊 테스트 결과 통계")
    
    accuracy = correct_count / len(test_cases) * 100
    print(f"  전체 정확도: {accuracy:.1f}% ({correct_count}/{len(test_cases)})")
    print()
    
    # 카테고리별 통계
    categories = {}
    for r in results:
        cat = r['result'].get('category', '없음')
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print("  감지된 카테고리:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  ├─ {cat}: {count}건")
    print()
    
    # 평균 처리 시간
    avg_time = sum(r['result']['processing_time'] for r in results) / len(results)
    print(f"  평균 처리 시간: {avg_time:.2f}초")
    print()
    
    # 오답 분석
    wrong_cases = [r for r in results if not r['correct']]
    if wrong_cases:
        print("  ⚠️  오답 케이스:")
        for r in wrong_cases:
            case = r['case']
            result = r['result']
            print(f"  ├─ \"{case['text'][:40]}...\"")
            print(f"  │   예상: {case['expected']}, 실제: {'부적절' if result['is_abusive'] else '정상'}")
            print(f"  │   점수: {result['abusive_score']:.2f}")
        print()
    
    print("=" * 70)
    
    # 결론
    print()
    print("📌 결론")
    print("-" * 70)
    print()
    
    if accuracy >= 90:
        print("  ✅ 우수: 프롬프트가 매우 효과적으로 작동합니다!")
    elif accuracy >= 70:
        print("  ⚠️  양호: 프롬프트가 대체로 잘 작동하지만 개선 여지가 있습니다.")
    else:
        print("  ❌ 미흡: 프롬프트 개선이 필요합니다.")
    
    print()
    print("  💡 개선된 프롬프트 특징:")
    print("  ├─ 욕설, 모욕, 성희롱, 위협 등 다양한 부적절 표현 감지")
    print("  ├─ 명시적 표현뿐만 아니라 문맥과 의도 파악")
    print("  ├─ 우회적이거나 비꼬는 표현도 감지 가능")
    print("  ├─ 카테고리 분류로 부적절 유형 식별")
    print("  └─ 점수로 심각도 수치화")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
