# -*- coding: utf-8 -*-
"""
KcBERT vs sLLM 비교 스크립트
"""

import sys
import os
import warnings
import time

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


def print_comparison():
    """비교 결과 출력"""
    print("\n" + "=" * 80)
    print("🔬 욕설 감지 시스템 비교: KcBERT vs sLLM")
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
    
    # KcBERT 초기화
    print("📥 KcBERT 모델 로딩 중...")
    from src.detector import AbusiveDetector
    
    class SuppressStderr:
        def __enter__(self):
            self._stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            return self
        def __exit__(self, *args):
            sys.stderr.close()
            sys.stderr = self._stderr
    
    with SuppressStderr():
        detector_kcbert = AbusiveDetector(threshold=0.5)
    
    print("✅ KcBERT 로딩 완료\n")
    
    # sLLM 초기화
    print("📥 sLLM 모델 로딩 중...")
    try:
        from src.detector_sllm import SLLMAbusiveDetector
        detector_sllm = SLLMAbusiveDetector(threshold=0.5)
    except ImportError as e:
        print(f"❌ sLLM 로딩 실패: {e}")
        print("   pip install llama-cpp-python 을 실행하세요.")
        return
    except FileNotFoundError as e:
        print(f"❌ 모델 파일 없음: {e}")
        return
    
    print("─" * 80)
    
    # 비교 테스트
    correct_kcbert = 0
    correct_sllm = 0
    total_time_kcbert = 0
    total_time_sllm = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {case['file']}")
        print(f"예상: {case['expected']}")
        print()
        print(f"텍스트: {case['text'][:70]}...")
        print()
        
        # KcBERT
        result_kcbert = detector_kcbert.predict(case['text'])
        detected_kcbert = "욕설" if result_kcbert['is_abusive'] else "정상"
        is_correct_kcbert = detected_kcbert == case['expected']
        total_time_kcbert += result_kcbert['processing_time']
        
        # sLLM
        result_sllm = detector_sllm.predict(case['text'])
        detected_sllm = "욕설" if result_sllm['is_abusive'] else "정상"
        is_correct_sllm = detected_sllm == case['expected']
        total_time_sllm += result_sllm['processing_time']
        
        # 결과 출력
        print("📊 결과 비교:")
        print()
        print(f"  {'항목':<20} {'KcBERT':<25} {'sLLM':<25}")
        print(f"  {'-' * 70}")
        print(f"  {'판정':<20} {detected_kcbert:<25} {detected_sllm:<25}")
        print(f"  {'점수':<20} {result_kcbert['abusive_score']:.4f}{'':<21} {result_sllm['abusive_score']:.4f}")
        print(f"  {'처리시간':<20} {result_kcbert['processing_time']:.3f}초{'':<18} {result_sllm['processing_time']:.3f}초")
        print(f"  {'정확도':<20} {'✅' if is_correct_kcbert else '❌':<25} {'✅' if is_correct_sllm else '❌'}")
        
        if result_sllm.get('reason'):
            print(f"\n  💡 sLLM 판단 이유: {result_sllm['reason']}")
        
        if is_correct_kcbert:
            correct_kcbert += 1
        if is_correct_sllm:
            correct_sllm += 1
        
        print()
        print("─" * 80)
    
    # 전체 요약
    total = len(test_cases)
    accuracy_kcbert = (correct_kcbert / total) * 100
    accuracy_sllm = (correct_sllm / total) * 100
    avg_time_kcbert = total_time_kcbert / total
    avg_time_sllm = total_time_sllm / total
    
    print()
    print("=" * 80)
    print("📈 전체 비교 결과")
    print("=" * 80)
    print()
    print(f"  {'지표':<20} {'KcBERT':<25} {'sLLM':<25}")
    print(f"  {'-' * 70}")
    print(f"  {'정확도':<20} {accuracy_kcbert:.1f}% ({correct_kcbert}/{total}){'':<12} {accuracy_sllm:.1f}% ({correct_sllm}/{total})")
    print(f"  {'평균 처리시간':<20} {avg_time_kcbert:.3f}초{'':<18} {avg_time_sllm:.3f}초")
    print(f"  {'총 처리시간':<20} {total_time_kcbert:.3f}초{'':<18} {total_time_sllm:.3f}초")
    print()
    
    # 승자 판정
    if accuracy_sllm > accuracy_kcbert:
        print(f"  🏆 승자: sLLM (+{accuracy_sllm - accuracy_kcbert:.1f}%p 더 정확)")
    elif accuracy_kcbert > accuracy_sllm:
        print(f"  🏆 승자: KcBERT (+{accuracy_kcbert - accuracy_sllm:.1f}%p 더 정확)")
    else:
        if avg_time_sllm < avg_time_kcbert:
            print(f"  🏆 승자: sLLM (동일 정확도, {avg_time_kcbert - avg_time_sllm:.3f}초 더 빠름)")
        else:
            print(f"  🏆 승자: KcBERT (동일 정확도, {avg_time_sllm - avg_time_kcbert:.3f}초 더 빠름)")
    
    print()
    print("=" * 80)
    
    # 특징 비교
    print()
    print("🎯 모델 특징 비교:")
    print()
    print("  KcBERT:")
    print("    ✅ 빠른 처리 속도")
    print("    ✅ 안정적인 성능")
    print("    ✅ GPU 가속 지원")
    print("    ❌ Fine-tuning 필요")
    print("    ❌ 문맥 이해 제한적")
    print()
    print("  sLLM:")
    print("    ✅ 우수한 문맥 이해")
    print("    ✅ 판단 이유 제공")
    print("    ✅ 온디바이스 실행")
    print("    ❌ 처리 시간 가변적")
    print("    ❌ 프롬프트 의존성")
    print()
    print("=" * 80)


if __name__ == "__main__":
    print_comparison()
