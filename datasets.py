import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import re
import unicodedata
import string
from underthesea import word_tokenize

class VietnameseNewsDataset(Dataset):
    def __init__(self, texts, author_labels, category_labels, tokenizer, 
                 max_len=200, remove_stopwords=False):
        """
        Args:
            texts: List các văn bản
            author_labels: Nhãn tác giả
            category_labels: Nhãn chủ đề
            tokenizer: Tokenizer để mã hóa văn bản
            max_len: Độ dài tối đa của văn bản
            remove_stopwords: True nếu muốn loại bỏ stopwords
        """
        self.texts = texts
        self.author_labels = author_labels
        self.category_labels = category_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.remove_stopwords = remove_stopwords
        
        # Đọc stopwords từ file
        if self.remove_stopwords:
            with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
                self.stopwords = set(f.read().splitlines())
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        # Chuyển về chữ thường
        text = text.lower()
        
        # Xóa số
        text = re.sub(r'\d+', '', text)
        
        # Xóa dấu câu
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Chuẩn hóa unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Tách từ
        tokens = word_tokenize(text)
        
        # Loại bỏ stopwords nếu được yêu cầu
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Nối lại thành câu
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tiền xử lý văn bản
        processed_text = self.preprocess_text(text)
        
        # Tokenize văn bản
        encoding = self.tokenizer.encode_plus(
            processed_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'author_labels': torch.tensor(self.author_labels[idx], dtype=torch.long),
            'category_labels': torch.tensor(self.category_labels[idx], dtype=torch.long)
        }