"""
개선된 욕설/폭언 감지 엔진 - 정확도 향상 버전
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any
from .model_loader import ModelLoader


class ImprovedAbusiveDetector:
    """개선된 KcBERT 기반 욕설/폭언 감지 엔진"""
    
    def __init__(self,
                 model_name: str = "beomi/kcbert-base",
                 cache_dir: str = "./models/kcbert",
                 threshold: float = 0.5,
                 max_length: int = 300,
                 use_dynamic_threshold: bool = True):
        """
        Args:
            model_name: 모델명
            cache_dir: 캐시 디렉토리
            threshold: 기본 감지 임계값 (동적 임계값 사용 시 기준값)
            max_length: 최대 토큰 길이
            use_dynamic_threshold: 동적 임계값 사용 여부
        """
        self.base_threshold = threshold
        self.max_length = max_length
        self.use_dynamic_threshold = use_dynamic_threshold
        
        # 모델 로더 초기화
        self.loader = ModelLoader(
            model_name=model_name,
            cache_dir=cache_dir
        )
        
        self.tokenizer = None
        self.model = None
        self.device = None
        
        # 강도별 욕설 패턴
        self.severe_patterns = {
            '씨발', '시발', 'ㅅㅂ', '병신', 'ㅂㅅ', '개새', '개새끼',
            '좆', '좃', '니미', '니엄마', '엿먹', '개같', '개 같',
            '미친새끼', '미친놈', '미친년', '지랄', '염병', '썅',
            '개자식', '개년', '개놈', '쓰레기새끼', '인간쓰레기'
        }
        
        self.moderate_patterns = {
            '짜증', '빡', '열받', '꺼져', '닥쳐', '엿같',
            '죽이고 싶', '때리고 싶', '작살', '개빡', 
            '미친', '미쳤', '돌았', '돌아버'
        }
        
        # 화이트리스트 (정상 표현)
        self.whitelist_patterns = {
            '답답하', '답답합니다', '아쉽', '안타깝', 
            '불편', '개선', '미친듯이 좋', '미친듯이 빠른',
            '죽이는 맛', '죽이는 디자인'
        }
        
        # 문맥 키워드 (문장 전체를 봐야 하는 경우)
        self.context_negative = {
            '답답': ['정말 답답', '너무 답답', '답답해 죽'],
            '미친': ['미친놈', '미친새끼', '미쳤어'],
        }
    
    def load_model(self):
        """모델 로드"""
        if self.model is None:
            print("\n" + "="*60)
            print("🤖 KcBERT 모델 초기화 중...")
            print("="*60 + "\n")
            
            self.tokenizer, self.model = self.loader.load()
            self.device = self.loader.get_device()
            
            print("\n" + "="*60)
            print("✅ 모델 초기화 완료!")
            print("="*60 + "\n")
    
    def _check_whitelist(self, text: str) -> bool:
        """화이트리스트 체크 (정상 표현인지)"""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.whitelist_patterns)
    
    def _check_context_negative(self, text: str, keyword: str) -> bool:
        """문맥상 부정적인지 확인"""
        if keyword not in self.context_negative:
            return False
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.context_negative[keyword])
    
    def _check_rule_based_advanced(self, text: str) -> Dict[str, Any]:
        """
        고급 규칙 기반 욕설 체크
        
        Returns:
            {
                'score': float,  # 0.0 ~ 1.0
                'severe_count': int,
                'moderate_count': int,
                'is_whitelist': bool
            }
        """
        text_lower = text.lower()
        
        # 화이트리스트 체크
        if self._check_whitelist(text):
            return {
                'score': 0.0,
                'severe_count': 0,
                'moderate_count': 0,
                'is_whitelist': True
            }
        
        severe_count = 0
        moderate_count = 0
        
        # 심각한 욕설 체크
        for pattern in self.severe_patterns:
            if pattern in text_lower:
                severe_count += 1
        
        # 중간 욕설 체크 (문맥 고려)
        for pattern in self.moderate_patterns:
            if pattern in text_lower:
                # 문맥 확인
                base_keyword = pattern.split()[0] if ' ' in pattern else pattern
                if base_keyword in self.context_negative:
                    if self._check_context_negative(text, base_keyword):
                        moderate_count += 1
                else:
                    moderate_count += 1
        
        # 점수 계산
        # 심각한 욕설: 개당 0.5점
        # 중간 욕설: 개당 0.25점
        score = min(severe_count * 0.5 + moderate_count * 0.25, 1.0)
        
        return {
            'score': score,
            'severe_count': severe_count,
            'moderate_count': moderate_count,
            'is_whitelist': False
        }
    
    def _calculate_dynamic_threshold(self, 
                                    rule_score: float, 
                                    model_score: float,
                                    confidence: float,
                                    rule_info: Dict) -> float:
        """
        동적 임계값 계산
        """
        if not self.use_dynamic_threshold:
            return self.base_threshold
        
        threshold = self.base_threshold
        
        # 규칙 기반 점수가 매우 높으면 (명확한 욕설)
        if rule_score >= 0.8:
            threshold = 0.35  # 낮은 임계값 (민감하게)
        
        # 규칙 기반 점수가 높으면
        elif rule_score >= 0.5:
            threshold = 0.4
        
        # 규칙 기반 점수가 매우 낮으면 (욕설 패턴 없음)
        elif rule_score < 0.1:
            threshold = 0.65  # 높은 임계값 (보수적으로)
        
        # 화이트리스트인 경우
        elif rule_info.get('is_whitelist'):
            threshold = 0.75  # 매우 높은 임계값
        
        # 신뢰도가 낮은 경우 보수적으로
        if confidence < 0.6:
            threshold += 0.1
        
        return min(threshold, 0.9)  # 최대 0.9
    
    def _adjust_final_score(self,
                           model_score: float,
                           rule_score: float,
                           confidence: float,
                           rule_info: Dict) -> float:
        """
        최종 점수 보정
        """
        # 화이트리스트인 경우 점수 대폭 감소
        if rule_info.get('is_whitelist'):
            return model_score * 0.3
        
        # 심각한 욕설이 있는 경우
        if rule_info['severe_count'] >= 2:
            # 규칙 점수를 더 높게 반영
            return model_score * 0.5 + rule_score * 0.5
        
        elif rule_info['severe_count'] >= 1:
            return model_score * 0.6 + rule_score * 0.4
        
        # 규칙과 모델이 일치하는 경우 (둘 다 높음)
        if model_score > 0.6 and rule_score > 0.6:
            # 확신 증가
            return min((model_score + rule_score) / 2 * 1.15, 1.0)
        
        # 규칙과 모델이 일치하는 경우 (둘 다 낮음)
        elif model_score < 0.3 and rule_score < 0.3:
            # 정상일 확률 높음
            return (model_score + rule_score) / 2 * 0.85
        
        # 불일치가 큰 경우
        elif abs(model_score - rule_score) > 0.5:
            # 보수적으로 (낮은 쪽 선택)
            return min(model_score, rule_score) * 1.1
        
        # 기본: 가중 평균
        # 규칙 점수가 높을수록 규칙의 가중치 증가
        rule_weight = 0.3 + (rule_score * 0.2)  # 0.3 ~ 0.5
        model_weight = 1.0 - rule_weight
        
        return model_score * model_weight + rule_score * rule_weight
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        개선된 단일 텍스트 예측
        """
        # 모델 로드
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # 1. 고급 규칙 기반 체크
        rule_info = self._check_rule_based_advanced(text)
        rule_score = rule_info['score']
        
        # 2. 모델 예측
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            abusive_prob = probabilities[0][1].item()
            confidence = torch.max(probabilities).item()
        
        # 3. 최종 점수 계산
        final_score = self._adjust_final_score(
            abusive_prob, rule_score, confidence, rule_info
        )
        
        # 4. 동적 임계값 계산
        threshold = self._calculate_dynamic_threshold(
            rule_score, abusive_prob, confidence, rule_info
        )
        
        # 5. 처리 시간
        processing_time = time.time() - start_time
        
        # 6. 결과 구성
        result = {
            "text": text,
            "is_abusive": final_score >= threshold,
            "confidence": confidence,
            "abusive_score": final_score,
            "model_score": abusive_prob,
            "rule_score": rule_score,
            "threshold": threshold,
            "processing_time": processing_time,
            "details": {
                "severe_words": rule_info['severe_count'],
                "moderate_words": rule_info['moderate_count'],
                "is_whitelist": rule_info['is_whitelist'],
                "dynamic_threshold_used": self.use_dynamic_threshold
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 예측"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_file(self, filepath: str) -> Dict[str, Any]:
        """파일에서 읽어서 예측"""
        from .preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        text = preprocessor.preprocess_file(filepath)
        
        result = self.predict(text)
        result["source_file"] = filepath
        
        return result
