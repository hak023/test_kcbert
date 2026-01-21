# -*- coding: utf-8 -*-
"""
기존 vs 개선 버전 비교 스크립트
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

# ⚡ Lazy import는 사용하지 않음 (비교를 위해 둘 다 필요)
from src.detector import AbusiveDetector
from src.detector_improved import ImprovedAbusiveDetector


def print_comparison():
    """비교 결과 출력"""
    print("\n" + "=" * 80)
    print("🔬 KcBERT 욕설 감지 시스템 - 기존 vs 개선 버전 비교")
    print("=" * 80 + "\n")
    
    # 테스트 케이스
    test_cases = [
        {
            "text": "고객: 야 거기 배송 왜 이렇게 느린거야? 이 병신들아. 씨발 빨리 안되냐고.",
            "expected": "욕설",
            "file": "abusive_call.txt"
        },
        {
            "text": "고객: 환불 좀 해주세요. 상품이 불량이에요. 정말 답답하네요.",
            "expected": "정상",
            "file": "complaint_call.txt"
        },
        {
            "text": "고객: 이미 일주일이 지났는데 내일이요? 진짜 너무한 거 아닙니까? 답답하네.",
            "expected": "정상",
            "file": "mixed_call.txt"
        },
        {
            "text": "고객: 안녕하세요. 제품 문의 드립니다. A 상품의 배송 기간이 궁금합니다.",
            "expected": "정상",
            "file": "normal_call.txt"
        },
    ]
    
    # 초기화 (stderr 숨기기)
    print("📥 모델 로딩 중...")
    
    class SuppressStderr:
        def __enter__(self):
            self._stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
        def __exit__(self, *args):
            sys.stderr.close()
            sys.stderr = self._stderr
    
    with SuppressStderr():
        detector_old = AbusiveDetector(threshold=0.5)
        detector_new = ImprovedAbusiveDetector(
            threshold=0.5,
            use_dynamic_threshold=True
        )
    
    print("✅ 모델 로딩 완료\n")
    print("─" * 80)
    
    # 각 테스트 케이스 비교
    correct_old = 0
    correct_new = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {case['file']}")
        print(f"예상: {case['expected']}")
        print()
        print(f"텍스트: {case['text'][:70]}...")
        print()
        
        # 기존 버전
        result_old = detector_old.predict(case['text'])
        detected_old = "욕설" if result_old['is_abusive'] else "정상"
        is_correct_old = detected_old == case['expected']
        
        # 개선 버전
        result_new = detector_new.predict(case['text'])
        detected_new = "욕설" if result_new['is_abusive'] else "정상"
        is_correct_new = detected_new == case['expected']
        
        # 결과 출력
        print("📊 결과 비교:")
        print()
        print(f"  {'항목':<20} {'기존 버전':<20} {'개선 버전':<20}")
        print(f"  {'-' * 60}")
        print(f"  {'판정':<20} {detected_old:<20} {detected_new:<20}")
        print(f"  {'최종 점수':<20} {result_old['abusive_score']:.4f}{'':<16} {result_new['abusive_score']:.4f}")
        print(f"  {'모델 점수':<20} {result_old['model_score']:.4f}{'':<16} {result_new['model_score']:.4f}")
        print(f"  {'규칙 점수':<20} {result_old['rule_score']:.4f}{'':<16} {result_new['rule_score']:.4f}")
        print(f"  {'임계값':<20} {result_old['threshold']:.4f}{'':<16} {result_new['threshold']:.4f}")
        print(f"  {'정확도':<20} {'✅' if is_correct_old else '❌':<20} {'✅' if is_correct_new else '❌'}")
        
        if result_new.get('details'):
            details = result_new['details']
            print()
            print(f"  🔍 상세 정보 (개선 버전):")
            print(f"     - 심각한 욕설: {details['severe_words']}개")
            print(f"     - 중간 욕설: {details['moderate_words']}개")
            print(f"     - 화이트리스트: {details['is_whitelist']}")
            print(f"     - 동적 임계값: {details['dynamic_threshold_used']}")
        
        if is_correct_old:
            correct_old += 1
        if is_correct_new:
            correct_new += 1
        
        print()
        print("─" * 80)
    
    # 전체 요약
    total = len(test_cases)
    accuracy_old = (correct_old / total) * 100
    accuracy_new = (correct_new / total) * 100
    
    print()
    print("=" * 80)
    print("📈 전체 정확도 비교")
    print("=" * 80)
    print()
    print(f"  기존 버전: {correct_old}/{total} = {accuracy_old:.1f}%")
    print(f"  개선 버전: {correct_new}/{total} = {accuracy_new:.1f}%")
    print()
    
    if accuracy_new > accuracy_old:
        improvement = accuracy_new - accuracy_old
        print(f"  ✨ 개선도: +{improvement:.1f}%p")
    elif accuracy_new == accuracy_old:
        print(f"  ➡️  동일한 정확도")
    else:
        print(f"  ⚠️  정확도 하락")
    
    print()
    print("=" * 80)
    
    # 개선 사항 요약
    print()
    print("🎯 적용된 개선 사항:")
    print()
    print("  1. ✅ 강도별 욕설 분류 (심각/중간)")
    print("  2. ✅ 화이트리스트 필터링 (정상 표현 보호)")
    print("  3. ✅ 문맥 기반 판단")
    print("  4. ✅ 동적 임계값 조정")
    print("  5. ✅ 스코어 보정 알고리즘")
    print()
    print("=" * 80)


if __name__ == "__main__":
    print_comparison()
