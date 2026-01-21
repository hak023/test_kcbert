"""
텍스트 전처리 모듈
"""

import re
import os
from typing import List


class TextPreprocessor:
    """통화 내용 텍스트 전처리 클래스"""
    
    def __init__(self, 
                 remove_special_chars: bool = True,
                 normalize_whitespace: bool = True):
        """
        Args:
            remove_special_chars: 특수문자 제거 여부
            normalize_whitespace: 공백 정규화 여부
        """
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace
    
    def load_from_file(self, filepath: str) -> str:
        """
        텍스트 파일에서 통화 내용 로드
        
        Args:
            filepath: 텍스트 파일 경로
            
        Returns:
            파일 내용 문자열
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
        
        # 다양한 인코딩 시도
        encodings = ['utf-8', 'cp949', 'euc-kr']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError(
            f"파일을 읽을 수 없습니다. 지원 인코딩: {encodings}"
        )
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 정제
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정제된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 공백 정규화
        if self.normalize_whitespace:
            # 연속된 공백을 하나로
            text = re.sub(r'\s+', ' ', text)
            # 줄바꿈 정규화
            text = re.sub(r'\n\s*\n', '\n', text)
        
        # 특수문자 제거 (선택적)
        if self.remove_special_chars:
            # 한글, 영문, 숫자, 기본 문장부호만 유지
            # 욕설 감지를 위해 너무 많이 제거하지 않도록 주의
            text = re.sub(r'[^\w\s가-힣.,!?:;\'"\-\(\)]', '', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """
        텍스트를 문장 단위로 분리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            문장 리스트
        """
        # 문장 종결 부호 기준 분리
        sentences = re.split(r'[.!?]\s+', text)
        
        # 빈 문장 제거
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def preprocess(self, text: str) -> str:
        """
        전체 전처리 파이프라인
        
        Args:
            text: 원본 텍스트
            
        Returns:
            전처리된 텍스트
        """
        return self.clean_text(text)
    
    def preprocess_file(self, filepath: str) -> str:
        """
        파일을 읽고 전처리
        
        Args:
            filepath: 파일 경로
            
        Returns:
            전처리된 텍스트
        """
        text = self.load_from_file(filepath)
        return self.preprocess(text)
