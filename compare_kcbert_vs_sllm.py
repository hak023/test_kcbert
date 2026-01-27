# -*- coding: utf-8 -*-
"""
KcBERT vs sLLM 성능 비교 스크립트
다양한 테스트 케이스로 두 모델의 성능 비교
"""

import sys
import os
import warnings
import time
from datetime import datetime
import json
from pathlib import Path

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


# 각 테스트 케이스의 예상 결과 (ground truth)
GROUND_TRUTH = {
    "test_01_normal_service.txt": {"label": "정상", "score": 0.1, "category": "없음"},
    "test_02_strong_complaint.txt": {"label": "경계선", "score": 0.4, "category": "불만"},
    "test_03_explicit_profanity.txt": {"label": "부적절", "score": 0.95, "category": "욕설"},
    "test_04_insult_no_swear.txt": {"label": "부적절", "score": 0.85, "category": "모욕"},
    "test_05_direct_threat.txt": {"label": "부적절", "score": 0.90, "category": "위협"},
    "test_06_sexual_harassment_direct.txt": {"label": "부적절", "score": 0.95, "category": "성희롱"},
    "test_07_sexual_harassment_subtle.txt": {"label": "부적절", "score": 0.75, "category": "성희롱"},
    "test_08_sarcastic_insult.txt": {"label": "부적절", "score": 0.70, "category": "모욕"},
    "test_09_borderline_angry.txt": {"label": "경계선", "score": 0.45, "category": "불만"},
    "test_10_borderline_frustrated.txt": {"label": "경계선", "score": 0.50, "category": "불만"},
    "test_11_threat_legal.txt": {"label": "경계선", "score": 0.55, "category": "위협"},
    "test_12_mixed_profanity_threat.txt": {"label": "부적절", "score": 0.98, "category": "복합"},
    "test_13_profanity_sexual.txt": {"label": "부적절", "score": 0.98, "category": "복합"},
    "test_14_polite_complaint.txt": {"label": "정상", "score": 0.2, "category": "없음"},
    "test_15_urgent_request.txt": {"label": "정상", "score": 0.1, "category": "없음"},
    "test_16_passive_aggressive.txt": {"label": "부적절", "score": 0.65, "category": "모욕"},
    "test_17_mild_insult.txt": {"label": "부적절", "score": 0.60, "category": "모욕"},
    "test_18_explicit_threat.txt": {"label": "부적절", "score": 0.95, "category": "위협"},
    "test_19_emotional_outburst.txt": {"label": "경계선", "score": 0.50, "category": "불만"},
    "test_20_appreciation.txt": {"label": "정상", "score": 0.0, "category": "없음"},
}


def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def classify_result(score, threshold=0.5):
    """점수로 레이블 분류"""
    if score < 0.3:
        return "정상"
    elif score < 0.6:
        return "경계선"
    else:
        return "부적절"


def calculate_accuracy(predictions, ground_truth):
    """정확도 계산"""
    correct = 0
    total = len(predictions)
    
    for filename, pred in predictions.items():
        if filename not in ground_truth:
            continue
        
        gt = ground_truth[filename]
        pred_label = classify_result(pred['score'])
        
        if pred_label == gt['label']:
            correct += 1
    
    return (correct / total * 100) if total > 0 else 0


def calculate_score_error(predictions, ground_truth):
    """평균 점수 오차 계산 (MAE)"""
    errors = []
    
    for filename, pred in predictions.items():
        if filename not in ground_truth:
            continue
        
        gt = ground_truth[filename]
        error = abs(pred['score'] - gt['score'])
        errors.append(error)
    
    return sum(errors) / len(errors) if errors else 0


def main():
    """메인 함수"""
    print_header("🔬 KcBERT vs sLLM 성능 비교 테스트")
    
    print("📝 테스트 개요")
    print("-" * 80)
    print("  ├─ 테스트 케이스: 20개")
    print("  ├─ 정상 케이스: 4개")
    print("  ├─ 경계선 케이스: 5개")
    print("  ├─ 부적절 케이스: 11개")
    print("  └─ 비교 모델: KcBERT vs sLLM")
    print()
    
    # 테스트 파일 확인
    samples_dir = Path("data/samples")
    test_files = sorted([f for f in samples_dir.glob("test_*.txt")])
    
    if not test_files:
        print("❌ 테스트 파일을 찾을 수 없습니다.")
        return
    
    print(f"✅ 테스트 파일 {len(test_files)}개 발견")
    print()
    
    # 모델 로딩
    print_header("1️⃣ KcBERT 모델 로딩")
    from src.detector_multi import MultiCategoryDetector
    
    kcbert_detector = MultiCategoryDetector()
    print("✅ KcBERT 모델 로딩 완료")
    
    print_header("2️⃣ sLLM 모델 로딩")
    from src.detector_sllm import SLLMAbusiveDetector
    
    sllm_detector = SLLMAbusiveDetector(verbose=False)
    print()
    
    # 테스트 실행
    print_header("3️⃣ 테스트 실행")
    
    kcbert_results = {}
    sllm_results = {}
    
    kcbert_total_time = 0
    sllm_total_time = 0
    
    for i, test_file in enumerate(test_files, 1):
        filename = test_file.name
        print(f"[{i}/{len(test_files)}] {filename}")
        print("-" * 80)
        
        # 파일 읽기
        with open(test_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"📝 내용: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print()
        
        # KcBERT 테스트
        print("  🔵 KcBERT 분석 중...", end=" ", flush=True)
        start_time = time.time()
        kcbert_result = kcbert_detector.predict(text)
        kcbert_time = time.time() - start_time
        kcbert_total_time += kcbert_time
        
        print(f"완료 ({kcbert_time:.2f}초)")
        print(f"     점수: {kcbert_result['abusive_score']:.3f}")
        print(f"     판정: {'부적절' if kcbert_result['is_abusive'] else '정상'}")
        print()
        
        kcbert_results[filename] = {
            'score': kcbert_result['abusive_score'],
            'is_abusive': kcbert_result['is_abusive'],
            'time': kcbert_time,
            'details': kcbert_result
        }
        
        # sLLM 테스트
        print("  🟢 sLLM 분석 중...", end=" ", flush=True)
        start_time = time.time()
        sllm_result = sllm_detector.predict(text)
        sllm_time = time.time() - start_time
        sllm_total_time += sllm_time
        
        print(f"완료 ({sllm_time:.2f}초)")
        print(f"     점수: {sllm_result['abusive_score']:.3f}")
        print(f"     판정: {'부적절' if sllm_result['is_abusive'] else '정상'}")
        print(f"     카테고리: {sllm_result.get('category', 'N/A')}")
        print()
        
        sllm_results[filename] = {
            'score': sllm_result['abusive_score'],
            'is_abusive': sllm_result['is_abusive'],
            'category': sllm_result.get('category', '없음'),
            'time': sllm_time,
            'details': sllm_result
        }
        
        print()
    
    # 통계 계산
    print_header("4️⃣ 성능 비교 결과")
    
    # 정확도
    kcbert_accuracy = calculate_accuracy(kcbert_results, GROUND_TRUTH)
    sllm_accuracy = calculate_accuracy(sllm_results, GROUND_TRUTH)
    
    # 점수 오차
    kcbert_mae = calculate_score_error(kcbert_results, GROUND_TRUTH)
    sllm_mae = calculate_score_error(sllm_results, GROUND_TRUTH)
    
    # 평균 처리 시간
    kcbert_avg_time = kcbert_total_time / len(test_files)
    sllm_avg_time = sllm_total_time / len(test_files)
    
    print("📊 전체 통계")
    print("-" * 80)
    print()
    
    print("  ⏱️  처리 시간 비교")
    print(f"     KcBERT: {kcbert_total_time:.2f}초 (평균 {kcbert_avg_time:.2f}초/건)")
    print(f"     sLLM:   {sllm_total_time:.2f}초 (평균 {sllm_avg_time:.2f}초/건)")
    print(f"     배속:   sLLM이 KcBERT보다 {sllm_avg_time/kcbert_avg_time:.1f}x {'느림' if sllm_avg_time > kcbert_avg_time else '빠름'}")
    print()
    
    print("  🎯 정확도 비교")
    print(f"     KcBERT: {kcbert_accuracy:.1f}%")
    print(f"     sLLM:   {sllm_accuracy:.1f}%")
    print(f"     차이:   {abs(sllm_accuracy - kcbert_accuracy):.1f}%p ({'sLLM 우수' if sllm_accuracy > kcbert_accuracy else 'KcBERT 우수'})")
    print()
    
    print("  📏 점수 오차 (MAE)")
    print(f"     KcBERT: {kcbert_mae:.3f}")
    print(f"     sLLM:   {sllm_mae:.3f}")
    print(f"     차이:   {abs(sllm_mae - kcbert_mae):.3f} ({'sLLM 우수' if sllm_mae < kcbert_mae else 'KcBERT 우수'})")
    print()
    
    # 카테고리별 분석
    print("  📂 sLLM 카테고리별 감지")
    category_count = {}
    for result in sllm_results.values():
        cat = result.get('category', '없음')
        category_count[cat] = category_count.get(cat, 0) + 1
    
    for cat, count in sorted(category_count.items(), key=lambda x: x[1], reverse=True):
        print(f"     ├─ {cat}: {count}건")
    print()
    
    # 상세 비교표
    print_header("5️⃣ 상세 비교표")
    
    print(f"{'파일명':<35} | {'실제':^8} | {'KcBERT':^8} | {'sLLM':^8} | {'카테고리':^10}")
    print("-" * 80)
    
    for filename in sorted(test_files, key=lambda x: x.name):
        fn = filename.name
        gt = GROUND_TRUTH.get(fn, {})
        kcbert = kcbert_results.get(fn, {})
        sllm = sllm_results.get(fn, {})
        
        gt_score = gt.get('score', 0)
        kcbert_score = kcbert.get('score', 0)
        sllm_score = sllm.get('score', 0)
        category = sllm.get('category', '-')
        
        print(f"{fn:<35} | {gt_score:>6.2f}  | {kcbert_score:>6.3f}  | {sllm_score:>6.3f}  | {category:^10}")
    
    print()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"data/results/comparison_kcbert_vs_sllm_{timestamp}.json"
    
    comparison_result = {
        "timestamp": timestamp,
        "test_count": len(test_files),
        "summary": {
            "kcbert": {
                "total_time": kcbert_total_time,
                "avg_time": kcbert_avg_time,
                "accuracy": kcbert_accuracy,
                "mae": kcbert_mae
            },
            "sllm": {
                "total_time": sllm_total_time,
                "avg_time": sllm_avg_time,
                "accuracy": sllm_accuracy,
                "mae": sllm_mae,
                "categories": category_count
            }
        },
        "kcbert_results": kcbert_results,
        "sllm_results": sllm_results,
        "ground_truth": GROUND_TRUTH
    }
    
    os.makedirs("data/results", exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 결과 저장: {result_file}")
    print()
    
    # 최종 결론
    print_header("6️⃣ 최종 결론")
    
    print("🏆 종합 평가")
    print("-" * 80)
    print()
    
    # 속도 우승자
    if kcbert_avg_time < sllm_avg_time:
        speed_winner = "KcBERT"
        speed_diff = f"{sllm_avg_time/kcbert_avg_time:.1f}x 빠름"
    else:
        speed_winner = "sLLM"
        speed_diff = f"{kcbert_avg_time/sllm_avg_time:.1f}x 빠름"
    
    # 정확도 우승자
    if kcbert_accuracy > sllm_accuracy:
        acc_winner = "KcBERT"
        acc_diff = f"{kcbert_accuracy - sllm_accuracy:.1f}%p 우수"
    else:
        acc_winner = "sLLM"
        acc_diff = f"{sllm_accuracy - kcbert_accuracy:.1f}%p 우수"
    
    print(f"  ⚡ 속도: {speed_winner} 승 ({speed_diff})")
    print(f"  🎯 정확도: {acc_winner} 승 ({acc_diff})")
    print(f"  📏 점수 정확성: {'sLLM' if sllm_mae < kcbert_mae else 'KcBERT'} 승")
    print()
    
    print("  💡 권장 사항")
    print("-" * 80)
    
    if kcbert_avg_time < sllm_avg_time * 0.5 and abs(kcbert_accuracy - sllm_accuracy) < 10:
        print("  ✅ 대량 처리: KcBERT (속도 우수)")
        print("  ✅ 정밀 분석: sLLM (카테고리 분류 가능)")
    elif sllm_accuracy > kcbert_accuracy + 10:
        print("  ✅ sLLM 권장: 정확도가 월등히 우수함")
    elif kcbert_avg_time < sllm_avg_time * 2:
        print("  ✅ KcBERT 권장: 속도와 정확도 균형")
    else:
        print("  ✅ 하이브리드 접근: 상황에 따라 선택")
        print("     - 실시간 처리: KcBERT")
        print("     - 배치 분석: sLLM")
        print("     - 의심 케이스: sLLM으로 재검증")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
