"""
KcBERT 모델 로더 모듈
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple


class ModelLoader:
    """KcBERT 모델 및 토크나이저 로더"""
    
    def __init__(self, 
                 model_name: str = "beomi/kcbert-base",
                 cache_dir: str = "./models/kcbert",
                 device: str = None):
        """
        Args:
            model_name: Hugging Face 모델명
            cache_dir: 모델 캐시 디렉토리
            device: 실행 디바이스 ('cuda', 'cpu', None=자동감지)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 캐시 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        토크나이저 로드
        
        Returns:
            KcBERT 토크나이저
        """
        if self.tokenizer is None:
            print(f"📥 토크나이저 로딩 중: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            print(f"✓ 토크나이저 로딩 완료")
        
        return self.tokenizer
    
    def load_model(self) -> AutoModelForSequenceClassification:
        """
        모델 로드
        
        Returns:
            KcBERT 모델
        """
        if self.model is None:
            print(f"📥 모델 로딩 중: {self.model_name}")
            print(f"   디바이스: {self.device}")
            
            # KcBERT는 기본적으로 사전학습만 된 상태
            # 실제로는 욕설 감지용으로 fine-tuning된 모델이 필요하지만,
            # 여기서는 마스크드 언어 모델을 사용하여 텍스트의 공격성을 추정
            
            # 참고: 실제 운영 환경에서는 fine-tuning된 모델 사용 필요
            try:
                from transformers import BertForSequenceClassification, BertConfig
                
                # KcBERT의 설정을 로드
                config = BertConfig.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                
                # 분류 레이어 추가
                config.num_labels = 2
                
                # 모델 로드 (ignore_mismatched_sizes로 크기 불일치 무시)
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_name,
                    config=config,
                    cache_dir=self.cache_dir,
                    ignore_mismatched_sizes=True  # 크기 불일치 무시
                )
                
                print("   ⚠️  기본 KcBERT 사용 (fine-tuning 안됨)")
                print("   💡 실제 사용을 위해서는 욕설 데이터로 fine-tuning 필요")
                
            except Exception as e:
                print(f"   ❌ 모델 로드 실패: {e}")
                raise
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ 모델 로딩 완료")
        
        return self.model
    
    def load(self) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        토크나이저와 모델 동시 로드
        
        Returns:
            (토크나이저, 모델) 튜플
        """
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        
        return tokenizer, model
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device
