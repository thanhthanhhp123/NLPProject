from datasets import *
from models import *
import pandas as pd
import torch
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# Add a new parameter to the function signature
def train_and_evaluate_model(
    model_type,
    train_data_path,
    output_dir='output',
    epochs=10,
    batch_size=32,
    gradient_accumulation_steps=1,  # Add this parameter
    learning_rate=5e-5,
    embedding_dim=300,
    hidden_dim=256,
    n_layers=2,
    dropout=0.5,
    max_len=256,
    bert_model_name=None,
    test_size=0.2,
    device=None,
    save_model=True,
    use_class_weights=True,
    use_cosine_scheduler=True
):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tạo thư mục output nếu chưa tồn tại
    model_output_dir = os.path.join(output_dir, model_type)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # Đọc dữ liệu
    print("Loading data...")
    df = pd.read_csv(train_data_path)

    # Filter top 20 authors by article count
    print("Filtering top 20 authors...")
    top_authors = df['author'].value_counts().nlargest(20).index.tolist()
    df = df[df['author'].isin(top_authors)]
    print(f"Dataset size after filtering: {len(df)} articles")

    print("Preparing labels...")
    author_encoder = LabelEncoder()
    category_encoder = LabelEncoder()

    df['author_encoded'] = author_encoder.fit_transform(df['author'])
    df['category_encoded'] = category_encoder.fit_transform(df['category'])

    n_authors = len(author_encoder.classes_)
    n_categories = len(category_encoder.classes_)

    print(f"Number of authors: {n_authors}")
    print(f"Number of categories: {n_categories}")

    print("Saving label encoders...")
    with open(os.path.join(model_output_dir, 'author_encoder.pkl'), 'wb') as f:
        pickle.dump(author_encoder, f)
    with open(os.path.join(model_output_dir, 'category_encoder.pkl'), 'wb') as f:
        pickle.dump(category_encoder, f)
    
    # Calculate class weights if enabled
    if use_class_weights:
        print("Calculating class weights...")
        from sklearn.utils.class_weight import compute_class_weight
        author_weights = compute_class_weight('balanced', classes=np.unique(df['author_encoded']), 
                                           y=df['author_encoded'])
        category_weights = compute_class_weight('balanced', classes=np.unique(df['category_encoded']), 
                                             y=df['category_encoded'])
        
        author_weights = torch.tensor(author_weights, dtype=torch.float).to(device)
        category_weights = torch.tensor(category_weights, dtype=torch.float).to(device)
        
        author_criterion = nn.CrossEntropyLoss(weight=author_weights)
        category_criterion = nn.CrossEntropyLoss(weight=category_weights)
    else:
        author_criterion = nn.CrossEntropyLoss()
        category_criterion = nn.CrossEntropyLoss()
    
    texts = df['text'].tolist()
    author_labels = df['author_encoded'].tolist()
    category_labels = df['category_encoded'].tolist()
    
    # Data Augmentation function
    def augment_text(text):
        words = text.split()
        if len(words) > 10:
            # Randomly drop 10% of words
            dropout_idx = random.sample(range(len(words)), int(len(words) * 0.1))
            augmented = [w for i, w in enumerate(words) if i not in dropout_idx]
            return ' '.join(augmented)
        return text

    # Apply augmentation to 30% of training samples
    augmented_texts = []
    augmented_author_labels = []
    augmented_category_labels = []
    
    aug_indices = random.sample(range(len(texts)), int(len(texts) * 0.3))
    for i in aug_indices:
        augmented_texts.append(augment_text(texts[i]))
        augmented_author_labels.append(author_labels[i])
        augmented_category_labels.append(category_labels[i])
    
    # Add augmented data
    texts.extend(augmented_texts)
    author_labels.extend(augmented_author_labels)
    category_labels.extend(augmented_category_labels)
    
    print(f"Dataset size after augmentation: {len(texts)}")

    # Split data into train/test sets
    train_texts, test_texts, train_author_labels, test_author_labels, train_category_labels, test_category_labels = train_test_split(
        texts, author_labels, category_labels, test_size=test_size, random_state=42, stratify=author_labels
    )
    
    if model_type in ['lstm', 'bilstm', 'enhanced_bilstm']:
        print("Building vocabulary...")
        word_to_idx = build_vocab(train_texts)  # Only build vocab from training data
        
        print(f"Vocabulary size: {len(word_to_idx)}")
        
        # Lưu từ điển
        with open(os.path.join(model_output_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(word_to_idx, f)
        
        # Khởi tạo dataset
        train_dataset = VnExpressDatasetLSTM(train_texts, train_author_labels, train_category_labels, word_to_idx, max_len)
        test_dataset = VnExpressDatasetLSTM(test_texts, test_author_labels, test_category_labels, word_to_idx, max_len)
    else:  # BERT models
        print(f"Loading tokenizer: {bert_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # Lưu tokenizer
        tokenizer.save_pretrained(os.path.join(model_output_dir, 'tokenizer'))
        
        # Khởi tạo dataset
        train_dataset = VnExpressDatasetBERT(train_texts, train_author_labels, train_category_labels, tokenizer, max_len)
        test_dataset = VnExpressDatasetBERT(test_texts, test_author_labels, test_category_labels, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Fix the model initialization part
    print(f"Initializing {model_type} model...")
    if model_type == 'enhanced_bilstm':
        model = EnhancedBiLSTMModel(
            vocab_size=len(word_to_idx),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_authors=n_authors,
            n_categories=n_categories,
            n_layers=n_layers,
            dropout=dropout
        )
    elif model_type == 'bert':
        model = MultiTaskBERTModel(n_authors=n_authors, n_categories=n_categories, dropout=dropout)
    elif model_type == 'phobert':
        model = MultiTaskPhoBERTModel(n_authors=n_authors, n_categories=n_categories, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Choose scheduler based on parameter
    if use_cosine_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,              # Restart after 5 epochs
            T_mult=2,           # Double restart interval after each restart
            eta_min=learning_rate * 1e-2  # Minimum learning rate
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': [],
        'author_accuracy': [],
        'category_accuracy': [],
        'author_f1': [],
        'category_f1': []
    }

    best_val_loss = float('inf')
    best_author_f1 = 0
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    print("Training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_author_loss = 0.0
        train_category_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Đưa dữ liệu vào device
            if model_type in ['lstm', 'bilstm', 'enhanced_bilstm']:
                input_ids = batch['input_ids'].to(device)
                author_labels = batch['author_label'].to(device)
                category_labels = batch['category_label'].to(device)
                
                # Forward pass
                author_logits, category_logits = model(input_ids)
            else:  # BERT models
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                author_labels = batch['author_label'].to(device)
                category_labels = batch['category_label'].to(device)
                
                # Forward pass
                author_logits, category_logits = model(input_ids, attention_mask)
            
            # Tính loss
            author_loss = author_criterion(author_logits, author_labels)
            category_loss = category_criterion(category_logits, category_labels)
            
            # Weighted sum of losses - give more importance to author classification
            loss = 0.7 * author_loss + 0.3 * category_loss
            
            # Scale the loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Only update weights after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            # Cập nhật loss (use full loss for logging)
            full_loss = loss.item() * gradient_accumulation_steps
            train_loss += full_loss
            train_author_loss += author_loss.item()
            train_category_loss += category_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(), 
                'author_loss': author_loss.item(), 
                'category_loss': category_loss.item()
            })
        
        # Update learning rate for cosine scheduler
        if use_cosine_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.2e}")
        
        # Tính loss trung bình cho epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_author_loss = train_author_loss / len(train_loader)
        avg_train_category_loss = train_category_loss / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation
        model.eval()
        val_loss = 0.0
        val_author_loss = 0.0
        val_category_loss = 0.0
        all_author_preds = []
        all_category_preds = []
        all_author_labels = []
        all_category_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Đưa dữ liệu vào device
                if model_type in ['lstm', 'bilstm', 'enhanced_bilstm']:
                    input_ids = batch['input_ids'].to(device)
                    author_labels = batch['author_label'].to(device)
                    category_labels = batch['category_label'].to(device)
                    
                    # Forward pass
                    author_logits, category_logits = model(input_ids)
                else:  # BERT models
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    author_labels = batch['author_label'].to(device)
                    category_labels = batch['category_label'].to(device)
                    
                    # Forward pass
                    author_logits, category_logits = model(input_ids, attention_mask)
                
                # Tính loss
                author_loss = author_criterion(author_logits, author_labels)
                category_loss = category_criterion(category_logits, category_labels)
                loss = 0.7 * author_loss + 0.3 * category_loss
                
                # Cập nhật loss
                val_loss += loss.item()
                val_author_loss += author_loss.item()
                val_category_loss += category_loss.item()
                
                # Lấy predicted labels
                author_preds = torch.argmax(author_logits, dim=1).cpu().numpy()
                category_preds = torch.argmax(category_logits, dim=1).cpu().numpy()
                
                # Thu thập predictions và labels
                all_author_preds.extend(author_preds)
                all_category_preds.extend(category_preds)
                all_author_labels.extend(author_labels.cpu().numpy())
                all_category_labels.extend(category_labels.cpu().numpy())
        
        # Tính metrics
        avg_val_loss = val_loss / len(test_loader)
        avg_val_author_loss = val_author_loss / len(test_loader)
        avg_val_category_loss = val_category_loss / len(test_loader)
        
        author_accuracy = accuracy_score(all_author_labels, all_author_preds)
        category_accuracy = accuracy_score(all_category_labels, all_category_preds)
        author_f1 = f1_score(all_author_labels, all_author_preds, average='weighted')
        category_f1 = f1_score(all_category_labels, all_category_preds, average='weighted')
        
        # Lưu vào history
        history['val_loss'].append(avg_val_loss)
        history['author_accuracy'].append(author_accuracy)
        history['category_accuracy'].append(category_accuracy)
        history['author_f1'].append(author_f1)
        history['category_f1'].append(category_f1)
        
        # Update LR scheduler if using ReduceLROnPlateau
        if not use_cosine_scheduler:
            scheduler.step(avg_val_loss)

        # In metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} (Author: {avg_train_author_loss:.4f}, Category: {avg_train_category_loss:.4f})")
        print(f"Val Loss: {avg_val_loss:.4f} (Author: {avg_val_author_loss:.4f}, Category: {avg_val_category_loss:.4f})")
        print(f"Author Accuracy: {author_accuracy:.4f}, F1: {author_f1:.4f}")
        print(f"Category Accuracy: {category_accuracy:.4f}, F1: {category_f1:.4f}")

        # Save based on validation loss
        if avg_val_loss < best_val_loss and save_model:
            best_val_loss = avg_val_loss
            print(f"Val Loss improved from {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(model_output_dir, f'best_val_loss_{model_type}_model.pt'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Save based on author F1 score
        if author_f1 > best_author_f1 and save_model:
            best_author_f1 = author_f1
            print(f"Author F1 improved to {best_author_f1:.4f}. Saving model...")
            torch.save(model.state_dict(), os.path.join(model_output_dir, f'best_f1_{model_type}_model.pt'))
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
    if save_model:
        torch.save(model.state_dict(), os.path.join(model_output_dir, f'last_{model_type}_model.pt'))

    # Print detailed classification reports
    print("\nClassification Report for Authors:")
    author_report = classification_report(
        all_author_labels, 
        all_author_preds, 
        target_names=author_encoder.classes_,
        output_dict=True
    )
    author_df = pd.DataFrame(author_report).transpose()
    print(author_df)
    author_df.to_csv(os.path.join(model_output_dir, 'author_classification_report.csv'))
    
    print("\nClassification Report for Categories:")
    category_report = classification_report(
        all_category_labels, 
        all_category_preds, 
        target_names=category_encoder.classes_,
        output_dict=True
    )
    category_df = pd.DataFrame(category_report).transpose()
    print(category_df)
    category_df.to_csv(os.path.join(model_output_dir, 'category_classification_report.csv'))
    
    # Vẽ training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title(f'{model_type} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['author_accuracy'], label='author')
    plt.plot(history['category_accuracy'], label='category')
    plt.title(f'{model_type} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['author_f1'], label='author')
    plt.plot(history['category_f1'], label='category')
    plt.title(f'{model_type} - F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Confusion matrix for authors (top 10)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    if len(author_encoder.classes_) <= 10:
        author_classes = author_encoder.classes_
    else:
        # Get top 10 authors by count
        top_indices = np.bincount(all_author_labels).argsort()[-10:]
        author_classes = [author_encoder.classes_[i] for i in top_indices]
        
        # Filter predictions for these authors only
        mask = np.isin(all_author_labels, top_indices)
        filtered_preds = np.array(all_author_preds)[mask]
        filtered_labels = np.array(all_author_labels)[mask]
        
        # Create confusion matrix
        plt.subplot(2, 2, 4)
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_indices, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=[author_encoder.classes_[i] for i in top_indices],
                   yticklabels=[author_encoder.classes_[i] for i in top_indices])
        plt.title('Top 10 Authors - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_output_dir, f'{model_type}_training_curves.png'))
    
    # Save history
    with open(os.path.join(model_output_dir, f'{model_type}_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Training completed for {model_type}. Results saved to {model_output_dir}")
    return model, history


# Hàm main để chạy tất cả các mô hình
if __name__ == "__main__":
    set_seed(42)
    
    # Đường dẫn dữ liệu
    data_path = 'data/all_articles.csv'
    
    # Cấu hình chung cho tất cả mô hình
    # Update the common parameters
    common_params = {
        'train_data_path': data_path,
        'epochs': 20,  # Reduce epochs to avoid memory issues
        'test_size': 0.2,
        'save_model': True,
        'use_class_weights': True,
        'use_cosine_scheduler': True,
        'gradient_accumulation_steps': 4,  # Add accumulation steps
    }

    # And then specific parameters for each model
    models_to_train = [
        {
            'model_type': 'enhanced_bilstm',
            'output_dir': 'output',
            'batch_size': 8,
            'max_len': 256,  # LSTM can handle longer sequences
            'learning_rate': 5e-4,
            'embedding_dim': 400,
            'hidden_dim': 512,
            'n_layers': 3,
            'dropout': 0.6
        },
        {
            'model_type': 'phobert',
            'output_dir': 'output',
            'batch_size': 4,
            'max_len': 128,  # Reduce from 256 to 128 for transformers
            'learning_rate': 2e-5,
            'bert_model_name': 'vinai/phobert-base',
            'dropout': 0.3
        }
    ]
    
    # Chạy lần lượt từng mô hình
    results = {}
    for model_config in models_to_train:
        print(f"\n{'='*50}")
        print(f"TRAINING MODEL: {model_config['model_type']}")
        print(f"{'='*50}\n")
        
        # Kết hợp cấu hình chung và riêng
        config = {**common_params, **model_config}
        
        # Train mô hình
        model, history = train_and_evaluate_model(**config)
        
        # Lưu kết quả
        results[model_config['model_type']] = {
            'history': history,
            'final_author_accuracy': history['author_accuracy'][-1],
            'final_category_accuracy': history['category_accuracy'][-1],
            'final_author_f1': history['author_f1'][-1],
            'final_category_f1': history['category_f1'][-1],
            'best_author_f1': max(history['author_f1']),
            'best_category_f1': max(history['category_f1'])
        }
    
    # So sánh kết quả của các mô hình
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Model': [],
        'Author Accuracy': [],
        'Author F1': [],
        'Category Accuracy': [],
        'Category F1': [],
        'Best Author F1': [],
        'Best Category F1': []
    })
    
    for model_type, result in results.items():
        new_row = {
            'Model': model_type,
            'Author Accuracy': f"{result['final_author_accuracy']:.4f}",
            'Author F1': f"{result['final_author_f1']:.4f}",
            'Category Accuracy': f"{result['final_category_accuracy']:.4f}",
            'Category F1': f"{result['final_category_f1']:.4f}",
            'Best Author F1': f"{result['best_author_f1']:.4f}",
            'Best Category F1': f"{result['best_category_f1']:.4f}"
        }
        comparison_df = pd.concat([comparison_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv('output/model_comparison.csv', index=False)
    
    print("\nTraining and evaluation completed for all models!")