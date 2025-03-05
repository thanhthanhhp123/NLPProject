import torch
import torch.nn as nn
from transformers import AutoModel

class LSTMMultiTask(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_authors, num_categories, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Tăng số lớp và thêm normalization
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Cải thiện task-specific layers
        self.author_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_authors)
        )
        
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories)
        )
    
    def forward(self, input_ids, attention_mask):
        # Embedding layer
        embedded = self.embedding(input_ids)
        
        # Apply attention mask
        embedded = embedded * attention_mask.unsqueeze(-1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)
        
        # Get the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Task-specific predictions
        author_out = self.author_classifier(last_hidden)
        category_out = self.category_classifier(last_hidden)
        
        return author_out, category_out

class BiLSTMMultiTask(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_authors, num_categories, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,  # Divide by 2 as bidirectional will double it
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific layers
        self.author_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_authors)
        )
        
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_categories)
        )
    
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = embedded * attention_mask.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(embedded)
        # Concatenate forward and backward states
        last_hidden = torch.cat((lstm_out[:, -1, :self.lstm.hidden_size],
                               lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
        
        author_out = self.author_classifier(last_hidden)
        category_out = self.category_classifier(last_hidden)
        
        return author_out, category_out

class PhoBERTMultiTask(nn.Module):
    def __init__(self, num_authors, num_categories, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        hidden_size = self.bert.config.hidden_size
        
        # Task-specific layers
        self.author_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_authors)
        )
        
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_categories)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Task-specific predictions
        author_out = self.author_classifier(pooled_output)
        category_out = self.category_classifier(pooled_output)
        
        return author_out, category_out

# Loss function
class MultiTaskLoss(nn.Module):
    def __init__(self, tasks_weights={'author': 0.5, 'category': 0.5}):
        super().__init__()
        self.tasks_weights = tasks_weights
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, author_pred, category_pred, author_true, category_true):
        author_loss = self.criterion(author_pred, author_true)
        category_loss = self.criterion(category_pred, category_true)
        
        # Weighted sum of losses
        total_loss = (self.tasks_weights['author'] * author_loss + 
                     self.tasks_weights['category'] * category_loss)
        
        return total_loss, author_loss, category_loss

class DynamicMultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        self.criterion = nn.CrossEntropyLoss()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, author_pred, category_pred, author_true, category_true):
        author_loss = self.criterion(author_pred, author_true)
        category_loss = self.criterion(category_pred, category_true)
        
        # Dynamic weight adjustment
        precision_1 = torch.exp(-self.log_vars[0])
        loss_1 = precision_1 * author_loss + self.log_vars[0]
        
        precision_2 = torch.exp(-self.log_vars[1])
        loss_2 = precision_2 * category_loss + self.log_vars[1]
        
        total_loss = loss_1 + loss_2
        
        return total_loss, author_loss, category_loss

if __name__ == "__main__":
    #save models architecture
    phobert_model = PhoBERTMultiTask(10, 5)
    torch.save(phobert_model, 'phobert_model.pth')

    lstm_model = LSTMMultiTask(10000, 300, 128, 10, 5)
    torch.save(lstm_model, 'lstm_model.pth')

    bilstm_model = BiLSTMMultiTask(10000, 300, 128, 10, 5)
    torch.save(bilstm_model, 'bilstm_model.pth')
    