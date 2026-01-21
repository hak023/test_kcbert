"""
욕설/폭언 감지 엔진 모듈
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any
from .model_loader import ModelLoader


class AbusiveDetector:
    """KcBERT 기반 욕설/폭언 감지 엔진"""
    
    def __init__(self,
                 model_name: str = "beomi/kcbert-base",
                 cache_dir: str = "./models/kcbert",
                 threshold: float = 0.5,
                 max_length: int = 300):  # KcBERT 최대 길이는 300
        """
        Args:
            model_name: 모델명
            cache_dir: 캐시 디렉토리
            threshold: 감지 임계값 (0.0 ~ 1.0)
            max_length: 최대 토큰 길이 (KcBERT는 300이 최대)
        """
        self.threshold = threshold
        self.max_length = max_length
        
        # 모델 로더 초기화
        self.loader = ModelLoader(
            model_name=model_name,
            cache_dir=cache_dir
        )
        
        # 모델과 토크나이저는 지연 로딩
        self.tokenizer = None
        self.model = None
        self.device = None
        
        # 간단한 규칙 기반 욕설 패턴 (보조 기능)
        # 실제로는 더 정교한 사전이 필요하지만, 예제용으로 간단히 구성
        self.abusive_patterns = [
            '시발', '씨발', '병신', '개새', '좆', '니미', 
            '지랄', '엿먹', '꺼져', 'ㅅㅂ', 'ㅂㅅ', '미친'
        ]
    
    def load_model(self):
        """모델 로드 (지연 로딩)"""
        if self.model is None:
            print("\n" + "="*60)
            print("🤖 KcBERT 모델 초기화 중...")
            print("="*60 + "\n")
            
            self.tokenizer, self.model = self.loader.load()
            self.device = self.loader.get_device()
            
            print("\n" + "="*60)
            print("✅ 모델 초기화 완료!")
            print("="*60 + "\n")
    
    def _check_rule_based(self, text: str) -> float:
        """
        규칙 기반 욕설 체크 (보조 기능)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            규칙 기반 점수 (0.0 ~ 1.0)
        """
        text_lower = text.lower()
        matches = sum(1 for pattern in self.abusive_patterns if pattern in text_lower)
        
        # 매칭된 패턴 수에 따라 점수 계산
        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.6
        elif matches == 2:
            return 0.8
        else:
            return 0.95
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        단일 텍스트 예측
        
        Args:
            text: 입력 텍스트
            
        Returns:
            감지 결과 딕셔너리
        """
        # 모델 로드 (처음 호출 시)
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 추론
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Softmax로 확률 계산
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            abusive_prob = probabilities[0][1].item()  # 욕설 클래스 확률
            confidence = torch.max(probabilities).item()
        
        # 규칙 기반 점수와 결합
        rule_score = self._check_rule_based(text)
        
        # 최종 점수 = (모델 점수 * 0.7) + (규칙 기반 점수 * 0.3)
        # 모델이 제대로 fine-tuning되지 않은 경우 규칙 기반에 더 의존
        if rule_score > 0.5:
            final_score = max(abusive_prob, rule_score)
        else:
            final_score = abusive_prob * 0.7 + rule_score * 0.3
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        # 결과 구성
        result = {
            "text": text,
            "is_abusive": final_score >= self.threshold,
            "confidence": confidence,
            "abusive_score": final_score,
            "model_score": abusive_prob,
            "rule_score": rule_score,
            "threshold": self.threshold,
            "processing_time": processing_time
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        배치 예측
        
        Args:
            texts: 입력 텍스트 리스트
            
        Returns:
            감지 결과 리스트
        """
        results = []
        
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def predict_file(self, filepath: str) -> Dict[str, Any]:
        """
        파일에서 읽어서 예측
        
        Args:
            filepath: 텍스트 파일 경로
            
        Returns:
            감지 결과
        """
        from .preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        text = preprocessor.preprocess_file(filepath)
        
        result = self.predict(text)
        result["source_file"] = filepath
        
        return result
