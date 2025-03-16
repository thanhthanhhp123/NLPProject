import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random

def build_vocab(texts, max_vocab_size=20000):
    word_counts = {}
    for text in texts:
        for word in text.split():
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # Sắp xếp từ theo tần suất xuất hiện
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Tạo từ điển với các từ phổ biến nhất
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_vocab_size-2]:
        word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


class VnExpressDatasetLSTM(Dataset):
    def __init__(self, texts, author_labels, category_labels, word_to_idx, max_len=256):
        self.texts = texts
        self.author_labels = author_labels
        self.category_labels = category_labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()[:self.max_len]
        
        # Chuyển từ thành indices
        input_ids = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Padding
        if len(input_ids) < self.max_len:
            input_ids = input_ids + [self.word_to_idx['<PAD>']] * (self.max_len - len(input_ids))
        
        return {
            'input_ids': torch.tensor(input_ids),
            'author_label': torch.tensor(self.author_labels[idx]),
            'category_label': torch.tensor(self.category_labels[idx])
        }
    
class VnExpressDatasetBERT(Dataset):
    def __init__(self, texts, author_labels, category_labels, tokenizer, max_len=256):
        self.texts = texts
        self.author_labels = author_labels
        self.category_labels = category_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'author_label': torch.tensor(self.author_labels[idx]),
            'category_label': torch.tensor(self.category_labels[idx])
        }

# Add to datasets.py or preprocessing step
def augment_text(text):
    # Simple synonym replacement or random word dropout
    words = text.split()
    if len(words) > 10:
        # Randomly drop 10% of words
        dropout_idx = random.sample(range(len(words)), int(len(words) * 0.1))
        augmented = [w for i, w in enumerate(words) if i not in dropout_idx]
        return ' '.join(augmented)
    return text