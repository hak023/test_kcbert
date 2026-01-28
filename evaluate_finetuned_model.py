# -*- coding: utf-8 -*-
"""
Fine-tuned 모델 평가 스크립트
원본 KcBERT vs Fine-tuned KcBERT 성능 비교
"""

import sys
import os
import warnings
import time
from pathlib import Path
from datetime import datetime
import json

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


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


# Ground Truth (실제 정답)
GROUND_TRUTH = {
    "test_01_normal_service.txt": {"label": "정상", "score": 0.1, "category": "없음"},
    "test_02_strong_complaint.txt": {"label": "경계선", "score": 0.4, "category": "불만"},
    "test_03_explicit_profanity.txt": {"label": "부적절", "score": 0.95, "category": "욕설"},
    "test_04_insult_no_swear.txt": {"label": "부적절", "score": 0.85, "category": "모욕"},
    "test_05_direct_threat.txt": {"label": "부적절", "score": 0.9, "category": "위협"},
    "test_06_sexual_harassment_direct.txt": {"label": "부적절", "score": 0.95, "category": "성희롱"},
    "test_07_sexual_harassment_subtle.txt": {"label": "부적절", "score": 0.75, "category": "성희롱"},
    "test_08_sarcastic_insult.txt": {"label": "부적절", "score": 0.7, "category": "모욕"},
    "test_09_borderline_angry.txt": {"label": "경계선", "score": 0.45, "category": "불만"},
    "test_10_borderline_frustrated.txt": {"label": "경계선", "score": 0.5, "category": "불만"},
    "test_11_threat_legal.txt": {"label": "경계선", "score": 0.55, "category": "위협"},
    "test_12_mixed_profanity_threat.txt": {"label": "부적절", "score": 0.98, "category": "복합"},
    "test_13_profanity_sexual.txt": {"label": "부적절", "score": 0.98, "category": "복합"},
    "test_14_polite_complaint.txt": {"label": "정상", "score": 0.2, "category": "없음"},
    "test_15_urgent_request.txt": {"label": "정상", "score": 0.1, "category": "없음"},
    "test_16_passive_aggressive.txt": {"label": "부적절", "score": 0.65, "category": "모욕"},
    "test_17_mild_insult.txt": {"label": "부적절", "score": 0.6, "category": "모욕"},
    "test_18_explicit_threat.txt": {"label": "부적절", "score": 0.95, "category": "위협"},
    "test_19_emotional_outburst.txt": {"label": "경계선", "score": 0.5, "category": "불만"},
    "test_20_appreciation.txt": {"label": "정상", "score": 0.0, "category": "없음"},
}


def main():
    """메인 함수"""
    print_header("🔬 Fine-tuned 모델 평가")
    
    print("📝 평가 개요")
    print("-" * 80)
    print("  ├─ 원본 KcBERT vs Fine-tuned KcBERT")
    print("  ├─ 테스트 케이스: 20개")
    print("  └─ 평가 지표: 정확도, MAE")
    print()
    
    # Fine-tuned 모델 확인
    finetuned_model_path = "models/kcbert-finetuned-issue-cases"
    if not os.path.exists(finetuned_model_path):
        print(f"❌ Fine-tuned 모델이 없습니다: {finetuned_model_path}")
        print("   먼저 'python finetune_issue_cases.py'를 실행하세요.")
        return
    
    print(f"✅ Fine-tuned 모델 발견: {finetuned_model_path}")
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
    print_header("1️⃣ 원본 KcBERT 모델 로딩")
    from src.detector_multi import MultiCategoryDetector
    
    original_detector = MultiCategoryDetector()
    print("✅ 원본 모델 로딩 완료")
    
    print_header("2️⃣ Fine-tuned KcBERT 모델 로딩")
    
    # Fine-tuned 모델 로더 (임시로 원본 사용, 모델 경로만 변경)
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_path,
        ignore_mismatched_sizes=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print("✅ Fine-tuned 모델 로딩 완료")
    print()
    
    # 워밍업 실행 (모델 초기화 시간 제외)
    print_header("2.5️⃣ 모델 워밍업")
    print("⏳ 모델 워밍업 중... (첫 케이스 처리 시간 보정을 위함)")
    print()
    
    warmup_text = "안녕하세요. 테스트입니다."
    
    print("  🔵 원본 KcBERT 워밍업...", end=" ", flush=True)
    _ = original_detector.predict(warmup_text)
    print("완료")
    
    print("  🟢 Fine-tuned KcBERT 워밍업...", end=" ", flush=True)
    warmup_inputs = tokenizer(
        warmup_text,
        add_special_tokens=True,
        max_length=300,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        _ = model(**warmup_inputs)
    print("완료")
    
    print()
    print("✅ 워밍업 완료! 이제 정확한 처리 시간 측정이 가능합니다.")
    
    # 테스트 실행
    print_header("3️⃣ 테스트 실행")
    
    original_results = {}
    finetuned_results = {}
    
    original_total_time = 0
    finetuned_total_time = 0
    
    for i, test_file in enumerate(test_files, 1):
        filename = test_file.name
        print(f"[{i}/{len(test_files)}] {filename}")
        print("-" * 80)
        
        # 파일 읽기
        with open(test_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"📝 내용: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print()
        
        # 원본 모델 테스트
        print("  🔵 원본 KcBERT 분석 중...", end=" ", flush=True)
        start_time = time.time()
        original_result = original_detector.predict(text)
        original_time = time.time() - start_time
        original_total_time += original_time
        
        print(f"완료 ({original_time:.2f}초)")
        print(f"     점수: {original_result['abusive_score']:.3f}")
        print(f"     판정: {'부적절' if original_result['is_abusive'] else '정상'}")
        print()
        
        original_results[filename] = {
            'score': original_result['abusive_score'],
            'is_abusive': original_result['is_abusive'],
            'time': original_time
        }
        
        # Fine-tuned 모델 테스트
        print("  🟢 Fine-tuned KcBERT 분석 중...", end=" ", flush=True)
        start_time = time.time()
        
        # 토큰화
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=300,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            score = probs[0][1].item()  # 부적절 확률
            is_abusive = score >= 0.5
        
        finetuned_time = time.time() - start_time
        finetuned_total_time += finetuned_time
        
        print(f"완료 ({finetuned_time:.2f}초)")
        print(f"     점수: {score:.3f}")
        print(f"     판정: {'부적절' if is_abusive else '정상'}")
        print()
        
        finetuned_results[filename] = {
            'score': score,
            'is_abusive': is_abusive,
            'time': finetuned_time
        }
        
        print()
    
    # 통계 계산
    print_header("4️⃣ 성능 비교 결과")
    
    # 정확도
    original_accuracy = calculate_accuracy(original_results, GROUND_TRUTH)
    finetuned_accuracy = calculate_accuracy(finetuned_results, GROUND_TRUTH)
    
    # 점수 오차
    original_mae = calculate_score_error(original_results, GROUND_TRUTH)
    finetuned_mae = calculate_score_error(finetuned_results, GROUND_TRUTH)
    
    # 평균 처리 시간
    original_avg_time = original_total_time / len(test_files)
    finetuned_avg_time = finetuned_total_time / len(test_files)
    
    print("📊 전체 통계")
    print("-" * 80)
    print()
    
    print("  ⏱️  처리 시간 비교")
    print(f"     원본:      {original_total_time:.2f}초 (평균 {original_avg_time:.2f}초/건)")
    print(f"     Fine-tuned: {finetuned_total_time:.2f}초 (평균 {finetuned_avg_time:.2f}초/건)")
    improvement = ((finetuned_avg_time - original_avg_time) / original_avg_time) * 100
    print(f"     변화:      {improvement:+.1f}%")
    print()
    
    print("  🎯 정확도 비교")
    print(f"     원본:      {original_accuracy:.1f}%")
    print(f"     Fine-tuned: {finetuned_accuracy:.1f}%")
    accuracy_improvement = finetuned_accuracy - original_accuracy
    print(f"     개선:      {accuracy_improvement:+.1f}%p {'✅' if accuracy_improvement > 0 else '⚠️'}")
    print()
    
    print("  📏 점수 오차 (MAE)")
    print(f"     원본:      {original_mae:.3f}")
    print(f"     Fine-tuned: {finetuned_mae:.3f}")
    mae_improvement = original_mae - finetuned_mae
    print(f"     개선:      {mae_improvement:+.3f} {'✅' if mae_improvement > 0 else '⚠️'}")
    print()
    
    # 상세 비교표
    print_header("5️⃣ 상세 비교표")
    
    print(f"{'파일명':<35} | {'실제':^8} | {'원본':^8} | {'Fine-tuned':^8} | {'개선':^8}")
    print("-" * 80)
    
    improvement_count = 0
    for filename in sorted(test_files, key=lambda x: x.name):
        fn = filename.name
        gt = GROUND_TRUTH.get(fn, {})
        orig = original_results.get(fn, {})
        fine = finetuned_results.get(fn, {})
        
        gt_score = gt.get('score', 0)
        orig_score = orig.get('score', 0)
        fine_score = fine.get('score', 0)
        
        # 개선 여부 판단
        orig_error = abs(orig_score - gt_score)
        fine_error = abs(fine_score - gt_score)
        improved = "✅" if fine_error < orig_error else ("⚠️" if fine_error > orig_error else "➖")
        
        if fine_error < orig_error:
            improvement_count += 1
        
        print(f"{fn:<35} | {gt_score:>6.2f}  | {orig_score:>6.3f}  | {fine_score:>6.3f}  | {improved:^8}")
    
    print()
    print(f"  개선된 케이스: {improvement_count}/{len(test_files)} ({improvement_count/len(test_files)*100:.1f}%)")
    print()
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"data/results/finetuned_evaluation_{timestamp}.json"
    
    evaluation_result = {
        "timestamp": timestamp,
        "test_count": len(test_files),
        "summary": {
            "original": {
                "total_time": original_total_time,
                "avg_time": original_avg_time,
                "accuracy": original_accuracy,
                "mae": original_mae
            },
            "finetuned": {
                "total_time": finetuned_total_time,
                "avg_time": finetuned_avg_time,
                "accuracy": finetuned_accuracy,
                "mae": finetuned_mae
            },
            "improvement": {
                "accuracy": accuracy_improvement,
                "mae": mae_improvement,
                "improved_cases": improvement_count,
                "improved_percentage": improvement_count / len(test_files) * 100
            }
        },
        "original_results": original_results,
        "finetuned_results": finetuned_results,
        "ground_truth": GROUND_TRUTH
    }
    
    os.makedirs("data/results", exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"💾 결과 저장: {result_file}")
    print()
    
    # 최종 결론
    print_header("6️⃣ 최종 결론")
    
    print("🏆 종합 평가")
    print("-" * 80)
    print()
    
    if accuracy_improvement > 10:
        print("  ✅ 우수: Fine-tuning이 매우 효과적이었습니다!")
        print(f"     정확도가 {accuracy_improvement:.1f}%p 향상되었습니다.")
    elif accuracy_improvement > 0:
        print("  ⚠️  양호: Fine-tuning이 어느 정도 효과가 있었습니다.")
        print(f"     정확도가 {accuracy_improvement:.1f}%p 향상되었습니다.")
    else:
        print("  ❌ 미흡: Fine-tuning 효과가 부족합니다.")
        print("     더 많은 학습 데이터가 필요할 수 있습니다.")
    
    print()
    
    if mae_improvement > 0.05:
        print("  ✅ 점수 정확성도 크게 개선되었습니다!")
    elif mae_improvement > 0:
        print("  ⚠️  점수 정확성이 약간 개선되었습니다.")
    else:
        print("  ❌ 점수 정확성 개선이 필요합니다.")
    
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
