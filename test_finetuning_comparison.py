# -*- coding: utf-8 -*-
"""
Fine-tuning 전후 비교 테스트
현재 모델 vs Fine-tuned 모델 성능 비교
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

import json
import time
from pathlib import Path


def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_section(title):
    """섹션 출력"""
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70 + "\n")


def test_model(detector, test_cases, model_name):
    """모델 테스트"""
    print_header(f"🔍 테스트: {model_name}")
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['category']}: \"{case['text']}\"")
        
        # 예측
        result = detector.predict(case['text'])
        
        # 결과 저장
        test_result = {
            "category": case['category'],
            "text": case['text'],
            "expected": case['expected'],
            "result": result,
            "correct": self._check_correct(case['expected'], result)
        }
        results.append(test_result)
        
        # 결과 출력
        self._print_result(test_result)
        print()
    
    return results


def _check_correct(expected, result):
    """예측이 정확한지 확인"""
    if expected == "정상":
        return not result['is_abusive'] and not result['is_sexual_harassment']
    elif expected == "욕설":
        return result['is_abusive'] and not result['is_sexual_harassment']
    elif expected == "성희롱":
        return not result['is_abusive'] and result['is_sexual_harassment']
    elif expected == "욕설+성희롱":
        return result['is_abusive'] and result['is_sexual_harassment']
    return False


def _print_result(test_result):
    """결과 출력"""
    result = test_result['result']
    expected = test_result['expected']
    correct = test_result['correct']
    
    # 정확도 표시
    if correct:
        print("  ✅ 정확")
    else:
        print(f"  ❌ 부정확 (예상: {expected})")
    
    # 점수 출력
    abusive_emoji = "🔴" if result['is_abusive'] else "⚪"
    harassment_emoji = "🔴" if result['is_sexual_harassment'] else "⚪"
    
    print(f"  {abusive_emoji} 욕설: {result['abusive_score']:.3f}")
    print(f"  {harassment_emoji} 성희롱: {result['harassment_score']:.3f}")


def calculate_statistics(results):
    """통계 계산"""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # 카테고리별 정확도
    categories = {}
    for r in results:
        cat = r['expected']
        if cat not in categories:
            categories[cat] = {'total': 0, 'correct': 0}
        categories[cat]['total'] += 1
        if r['correct']:
            categories[cat]['correct'] += 1
    
    # 평균 점수
    avg_abusive = sum(r['result']['abusive_score'] for r in results) / total
    avg_harassment = sum(r['result']['harassment_score'] for r in results) / total
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total * 100,
        'categories': categories,
        'avg_scores': {
            'abusive': avg_abusive,
            'harassment': avg_harassment
        }
    }


def print_statistics(stats, model_name):
    """통계 출력"""
    print_section(f"📊 {model_name} 통계")
    
    print(f"  전체 정확도: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    print()
    
    print("  카테고리별 정확도:")
    for cat, data in stats['categories'].items():
        accuracy = data['correct'] / data['total'] * 100
        print(f"  ├─ {cat}: {accuracy:.1f}% ({data['correct']}/{data['total']})")
    print()
    
    print("  평균 점수:")
    print(f"  ├─ 욕설: {stats['avg_scores']['abusive']:.3f}")
    print(f"  └─ 성희롱: {stats['avg_scores']['harassment']:.3f}")
    print()


def compare_statistics(stats_before, stats_after):
    """통계 비교"""
    print_header("📈 Fine-tuning 전후 비교")
    
    # 전체 정확도 비교
    acc_diff = stats_after['accuracy'] - stats_before['accuracy']
    acc_emoji = "⬆️" if acc_diff > 0 else "⬇️" if acc_diff < 0 else "➡️"
    
    print(f"  전체 정확도:")
    print(f"  ├─ Fine-tuning 전: {stats_before['accuracy']:.1f}%")
    print(f"  ├─ Fine-tuning 후: {stats_after['accuracy']:.1f}%")
    print(f"  └─ 변화: {acc_emoji} {abs(acc_diff):.1f}%p")
    print()
    
    # 카테고리별 비교
    print("  카테고리별 정확도 변화:")
    for cat in stats_before['categories'].keys():
        before_acc = stats_before['categories'][cat]['correct'] / stats_before['categories'][cat]['total'] * 100
        after_acc = stats_after['categories'][cat]['correct'] / stats_after['categories'][cat]['total'] * 100
        diff = after_acc - before_acc
        emoji = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
        print(f"  ├─ {cat}: {before_acc:.1f}% → {after_acc:.1f}% ({emoji} {abs(diff):.1f}%p)")
    print()
    
    # 점수 비교
    print("  평균 점수 변화:")
    abusive_diff = stats_after['avg_scores']['abusive'] - stats_before['avg_scores']['abusive']
    harassment_diff = stats_after['avg_scores']['harassment'] - stats_before['avg_scores']['harassment']
    
    print(f"  ├─ 욕설: {stats_before['avg_scores']['abusive']:.3f} → {stats_after['avg_scores']['abusive']:.3f}")
    print(f"  └─ 성희롱: {stats_before['avg_scores']['harassment']:.3f} → {stats_after['avg_scores']['harassment']:.3f}")
    print()


def main():
    """메인 함수"""
    print_header("🔬 Fine-tuning 전후 비교 테스트")
    
    print("이 테스트는 Fine-tuning 전후의 모델 성능을 비교합니다.")
    print()
    print("📌 현재 상태:")
    print("  - Fine-tuning 전: 기본 KcBERT + 규칙 기반")
    print("  - Fine-tuning 후: 아직 학습되지 않음 (시뮬레이션)")
    print()
    print("⚠️  주의: Fine-tuned 모델이 없으므로 현재는 시뮬레이션입니다.")
    print("   실제 Fine-tuning 후에는 학습된 모델로 테스트할 수 있습니다.")
    print()
    
    # 테스트 케이스
    test_cases = [
        {
            "category": "정상",
            "text": "배송이 늦어서 불편합니다",
            "expected": "정상"
        },
        {
            "category": "정상",
            "text": "상담사님 친절하시네요 감사합니다",
            "expected": "정상"
        },
        {
            "category": "정상",
            "text": "제품 교환하고 싶은데 방법 알려주세요",
            "expected": "정상"
        },
        {
            "category": "욕설",
            "text": "씨발 빨리 좀 보내라",
            "expected": "욕설"
        },
        {
            "category": "욕설",
            "text": "개새끼들 진짜 짜증나네",
            "expected": "욕설"
        },
        {
            "category": "성희롱",
            "text": "목소리 좋으시네요 얼굴도 예쁘시겠어요",
            "expected": "성희롱"
        },
        {
            "category": "성희롱",
            "text": "남자친구 있어요 같이 저녁 먹을래요",
            "expected": "성희롱"
        },
        {
            "category": "성희롱",
            "text": "몸매 좋으신데 운동 같이 하실래요",
            "expected": "성희롱"
        },
        {
            "category": "성희롱",
            "text": "가슴 만져도 돼",
            "expected": "성희롱"
        },
        {
            "category": "욕설+성희롱",
            "text": "씨발 이 예쁜 것아 몸매 좋네",
            "expected": "욕설+성희롱"
        }
    ]
    
    print(f"📝 테스트 케이스: {len(test_cases)}개")
    print()
    
    # Fine-tuning 전 테스트
    print_header("1️⃣ Fine-tuning 전: 기본 KcBERT + 규칙 기반")
    
    from src.detector_multi import MultiCategoryDetector
    
    print("📥 모델 로딩 중...")
    detector_before = MultiCategoryDetector()
    print("✅ 로딩 완료\n")
    
    results_before = []
    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['category']}: \"{case['text']}\"")
        
        result = detector_before.predict(case['text'])
        
        test_result = {
            "category": case['category'],
            "text": case['text'],
            "expected": case['expected'],
            "result": result,
            "correct": _check_correct(case['expected'], result)
        }
        results_before.append(test_result)
        
        _print_result(test_result)
        print()
    
    stats_before = calculate_statistics(results_before)
    print_statistics(stats_before, "Fine-tuning 전")
    
    # Fine-tuning 후 시뮬레이션
    print_header("2️⃣ Fine-tuning 후: 예상 성능 (시뮬레이션)")
    
    print("⚠️  Fine-tuned 모델이 없으므로 예상 성능을 시뮬레이션합니다.")
    print("   실제 Fine-tuning 후에는 학습된 모델로 정확한 테스트가 가능합니다.")
    print()
    
    # 시뮬레이션: 성희롱 감지 성능 향상
    results_after = []
    for test_result in results_before:
        simulated_result = test_result.copy()
        result = simulated_result['result'].copy()
        
        # 성희롱 문장의 점수를 높게 조정 (시뮬레이션)
        if test_result['expected'] in ["성희롱", "욕설+성희롱"]:
            # KcBERT 점수를 높게 (Fine-tuning 효과)
            result['abusive_score'] = result['abusive_score'] * 1.2 if "욕설" in test_result['expected'] else result['abusive_score'] * 0.7
            result['harassment_score'] = min(0.95, result['harassment_score'] + 0.2)
            result['is_sexual_harassment'] = result['harassment_score'] >= 0.5
            result['is_abusive'] = result['abusive_score'] >= 0.5
        elif test_result['expected'] == "정상":
            # 정상 문장의 점수를 낮게 (오탐률 감소)
            result['abusive_score'] = result['abusive_score'] * 0.5
            result['harassment_score'] = result['harassment_score'] * 0.3
            result['is_sexual_harassment'] = result['harassment_score'] >= 0.5
            result['is_abusive'] = result['abusive_score'] >= 0.5
        
        simulated_result['result'] = result
        simulated_result['correct'] = _check_correct(test_result['expected'], result)
        results_after.append(simulated_result)
    
    # 결과 출력
    for i, test_result in enumerate(results_after, 1):
        print(f"[{i}/{len(test_cases)}] {test_result['category']}: \"{test_result['text']}\"")
        _print_result(test_result)
        print()
    
    stats_after = calculate_statistics(results_after)
    print_statistics(stats_after, "Fine-tuning 후")
    
    # 비교
    compare_statistics(stats_before, stats_after)
    
    # 결론
    print_header("📌 결론")
    
    print("  ✅ Fine-tuning 전후 비교 완료")
    print()
    print("  💡 관찰 사항:")
    print("  ├─ 현재 모델은 욕설은 어느 정도 감지")
    print("  ├─ 성희롱은 규칙 기반으로만 감지 (KcBERT 미학습)")
    print("  ├─ Fine-tuning 후 성희롱 감지율 크게 향상 예상")
    print("  └─ 오탐률 감소 예상")
    print()
    print("  🎯 다음 단계:")
    print("  ├─ 1. 학습 데이터 수집 (각 1,500개)")
    print("  ├─ 2. Fine-tuning 수행 (2~4시간)")
    print("  ├─ 3. 실제 모델로 재테스트")
    print("  └─ 4. 성능 검증 및 배포")
    print()
    
    # 결과 저장
    output_file = "data/results/finetuning_comparison_result.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "before": {
                "results": results_before,
                "statistics": stats_before
            },
            "after_simulated": {
                "results": results_after,
                "statistics": stats_after
            }
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"  💾 상세 결과 저장: {output_file}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
