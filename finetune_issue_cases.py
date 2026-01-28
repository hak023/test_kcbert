# -*- coding: utf-8 -*-
"""
이슈 케이스 Fine-tuning 스크립트
KcBERT 모델을 이슈 케이스로 재학습
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


class AbusiveDataset(Dataset):
    """욕설/폭언 감지 데이터셋"""
    
    def __init__(self, texts, labels, tokenizer, max_length=300):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(pred):
    """평가 지표 계산"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_header(title):
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """메인 함수"""
    print_header("🔧 이슈 케이스 Fine-tuning")
    
    print("📝 작업 개요")
    print("-" * 70)
    print("  ├─ 목적: 테스트에서 실패한 이슈 케이스로 모델 개선")
    print("  ├─ 데이터: 20개 이슈 케이스")
    print("  ├─ 방법: KcBERT 모델 Fine-tuning")
    print("  └─ 평가: 정확도, Precision, Recall, F1")
    print()
    
    # 1. 데이터 로드
    print_header("1️⃣ 데이터 로드")
    
    data_path = "data/training/issue_cases_training.csv"
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일이 없습니다: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(df)}개")
    print()
    
    print("  데이터 분포:")
    print(f"  ├─ 부적절 (label=1): {sum(df['label'] == 1)}개")
    print(f"  └─ 정상 (label=0): {sum(df['label'] == 0)}개")
    print()
    
    # 2. 데이터 분할
    print_header("2️⃣ 데이터 분할")
    
    # 데이터가 적으므로 80:20 분할
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )
    
    print(f"  ├─ 학습 데이터: {len(train_texts)}개")
    print(f"  └─ 검증 데이터: {len(val_texts)}개")
    print()
    
    # 3. 토크나이저 및 모델 로드
    print_header("3️⃣ 모델 로드")
    
    model_name = "beomi/kcbert-base"
    print(f"  모델: {model_name}")
    print()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True
    )
    
    print("✅ 모델 로드 완료")
    print()
    
    # 4. 데이터셋 생성
    print_header("4️⃣ 데이터셋 생성")
    
    train_dataset = AbusiveDataset(train_texts, train_labels, tokenizer)
    val_dataset = AbusiveDataset(val_texts, val_labels, tokenizer)
    
    print(f"✅ 학습 데이터셋: {len(train_dataset)}개")
    print(f"✅ 검증 데이터셋: {len(val_dataset)}개")
    print()
    
    # 5. 학습 설정
    print_header("5️⃣ 학습 설정")
    
    output_dir = "models/kcbert-finetuned-issue-cases"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,  # 소량 데이터이므로 많은 에폭
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none"
    )
    
    print("  학습 설정:")
    print(f"  ├─ 에폭: {training_args.num_train_epochs}")
    print(f"  ├─ 배치 크기: {training_args.per_device_train_batch_size}")
    print(f"  ├─ Learning Rate: {training_args.learning_rate}")
    print(f"  └─ 출력 디렉토리: {output_dir}")
    print()
    
    # 6. Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 7. Fine-tuning 시작
    print_header("6️⃣ Fine-tuning 시작")
    
    print("⏳ 학습 중... (약 5-10분 소요)")
    print()
    
    train_result = trainer.train()
    
    print()
    print("✅ Fine-tuning 완료!")
    print()
    
    # 8. 모델 저장
    print_header("7️⃣ 모델 저장")
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 모델 저장 완료: {output_dir}")
    print()
    
    # 9. 평가
    print_header("8️⃣ 모델 평가")
    
    eval_result = trainer.evaluate()
    
    print("  검증 데이터 평가 결과:")
    print(f"  ├─ Accuracy:  {eval_result['eval_accuracy']:.4f}")
    print(f"  ├─ Precision: {eval_result['eval_precision']:.4f}")
    print(f"  ├─ Recall:    {eval_result['eval_recall']:.4f}")
    print(f"  └─ F1 Score:  {eval_result['eval_f1']:.4f}")
    print()
    
    # 10. 학습 기록 저장
    print_header("9️⃣ 학습 기록 저장")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"data/results/finetuning_result_{timestamp}.json"
    
    result = {
        "timestamp": timestamp,
        "model": model_name,
        "output_dir": output_dir,
        "training_data_size": len(train_texts),
        "validation_data_size": len(val_texts),
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "train_result": {
            "train_loss": float(train_result.training_loss),
            "train_runtime": train_result.metrics['train_runtime'],
            "train_samples_per_second": train_result.metrics['train_samples_per_second']
        },
        "eval_result": {
            "accuracy": float(eval_result['eval_accuracy']),
            "precision": float(eval_result['eval_precision']),
            "recall": float(eval_result['eval_recall']),
            "f1": float(eval_result['eval_f1']),
            "loss": float(eval_result['eval_loss'])
        }
    }
    
    os.makedirs("data/results", exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 학습 기록 저장: {result_file}")
    print()
    
    # 11. 최종 요약
    print_header("🎯 Fine-tuning 완료")
    
    print("  📊 최종 결과:")
    print(f"  ├─ 정확도: {eval_result['eval_accuracy']*100:.1f}%")
    print(f"  ├─ F1 Score: {eval_result['eval_f1']:.4f}")
    print(f"  └─ 모델 위치: {output_dir}")
    print()
    
    print("  🎯 다음 단계:")
    print("  ├─ 1. Fine-tuned 모델로 20개 케이스 재평가")
    print("  ├─ 2. 개선 효과 확인")
    print("  └─ 3. 필요시 추가 학습 데이터 수집")
    print()
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
