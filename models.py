import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskPhoBERTModel(nn.Module):
    def __init__(self, n_authors, n_categories, dropout=0.3):
        super(MultiTaskPhoBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base')
        
        # Freeze the BERT layers (optional, can be fine-tuned depending on dataset size)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        hidden_size = self.bert.config.hidden_size
        
        # More complex classification heads with dropout
        self.author_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_authors)
        )
        
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_categories)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Using CLS token representation or mean pooling
        # Option 1: CLS token
        pooled_output = outputs[1]
        
        # Option 2: Mean pooling (often better than CLS)
        # token_embeddings = outputs[0]  # First element is all token embeddings
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        author_logits = self.author_classifier(pooled_output)
        category_logits = self.category_classifier(pooled_output)
        return author_logits, category_logits

class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_authors, n_categories, n_layers=2, dropout=0.5):
        super(EnhancedBiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Stacked BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification heads
        self.fc_author = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_authors)
        )
        
        self.fc_category = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_categories)
        )
        
        self.dropout = nn.Dropout(dropout)

    def attention_mechanism(self, lstm_output):
        # Apply attention
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # Apply attention instead of just taking last hidden state
        context_vector = self.attention_mechanism(lstm_out)
        
        author_logits = self.fc_author(context_vector)
        category_logits = self.fc_category(context_vector)
        return author_logits, category_logits
