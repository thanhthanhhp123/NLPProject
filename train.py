import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
import seaborn as sns
from models import LSTMMultiTask, BiLSTMMultiTask, PhoBERTMultiTask, MultiTaskLoss, DynamicMultiTaskLoss
from datasets import VietnameseNewsDataset
import warnings
from torch.serialization import safe_globals, add_safe_globals
warnings.filterwarnings('ignore')

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, model_name, early_stopping_patience=10):
    # Lưu lịch sử training
    history = {
        'train_loss': [], 'val_loss': [],
        'train_author_acc': [], 'val_author_acc': [],
        'train_category_acc': [], 'val_category_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_author_correct = 0
        train_category_correct = 0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            author_labels = batch['author_labels'].to(device)
            category_labels = batch['category_labels'].to(device)
            
            optimizer.zero_grad()
            
            author_output, category_output = model(input_ids, attention_mask)
            loss, author_loss, category_loss = criterion(
                author_output, category_output,
                author_labels, category_labels
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            
            # Calculate accuracies
            author_pred = torch.argmax(author_output, dim=1)
            category_pred = torch.argmax(category_output, dim=1)
            
            train_author_correct += (author_pred == author_labels).sum().item()
            train_category_correct += (category_pred == category_labels).sum().item()
            total_samples += len(author_labels)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_author_correct = 0
        val_category_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                author_labels = batch['author_labels'].to(device)
                category_labels = batch['category_labels'].to(device)
                
                author_output, category_output = model(input_ids, attention_mask)
                loss, _, _ = criterion(
                    author_output, category_output,
                    author_labels, category_labels
                )
                
                val_losses.append(loss.item())
                
                author_pred = torch.argmax(author_output, dim=1)
                category_pred = torch.argmax(category_output, dim=1)
                
                val_author_correct += (author_pred == author_labels).sum().item()
                val_category_correct += (category_pred == category_labels).sum().item()
                val_total += len(author_labels)
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_author_acc = train_author_correct / total_samples
        train_category_acc = train_category_correct / total_samples
        val_author_acc = val_author_correct / val_total
        val_category_acc = val_category_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_author_acc'].append(train_author_acc)
        history['val_author_acc'].append(val_author_acc)
        history['train_category_acc'].append(train_category_acc)
        history['val_category_acc'].append(val_category_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Author Acc: {train_author_acc:.4f}, Val Author Acc: {val_author_acc:.4f}')
        print(f'Train Category Acc: {train_category_acc:.4f}, Val Category Acc: {val_category_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'checkpoints/{model_name}_best.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
    
    return history

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    author_correct = 0
    category_correct = 0
    total = 0
    
    author_preds = []
    author_trues = []
    category_preds = []
    category_trues = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            author_labels = batch['author_labels'].to(device)
            category_labels = batch['category_labels'].to(device)
            
            author_output, category_output = model(input_ids, attention_mask)
            loss, _, _ = criterion(author_output, category_output,
                                 author_labels, category_labels)
            
            test_loss += loss.item()
            
            author_pred = torch.argmax(author_output, dim=1)
            category_pred = torch.argmax(category_output, dim=1)
            
            author_correct += (author_pred == author_labels).sum().item()
            category_correct += (category_pred == category_labels).sum().item()
            total += len(author_labels)
            
            author_preds.extend(author_pred.cpu().numpy())
            author_trues.extend(author_labels.cpu().numpy())
            category_preds.extend(category_pred.cpu().numpy())
            category_trues.extend(category_labels.cpu().numpy())
    
    return {
        'test_loss': test_loss / len(test_loader),
        'author_accuracy': author_correct / total,
        'category_accuracy': category_correct / total,
        'author_predictions': author_preds,
        'author_true': author_trues,
        'category_predictions': category_preds,
        'category_true': category_trues
    }

def plot_training_history(history, model_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot author accuracy
    ax2.plot(history['train_author_acc'], label='Train Author Acc')
    ax2.plot(history['val_author_acc'], label='Val Author Acc')
    ax2.set_title('Author Classification Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Plot category accuracy
    ax3.plot(history['train_category_acc'], label='Train Category Acc')
    ax3.plot(history['val_category_acc'], label='Val Category Acc')
    ax3.set_title('Category Classification Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png')
    plt.close()

def load_checkpoint(model_path, model, device):
    """Safely load model checkpoint"""
    try:
        # Add numpy scalar to safe globals
        with safe_globals(["numpy.core.multiarray.scalar"]):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning: Failed to load checkpoint with weights_only=True. Trying alternative method...")
        try:
            # Try loading without weights_only
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Continuing with current model state...")
            return None
    return checkpoint

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj

def save_results(results, filename):
    """Save results with proper type conversion"""
    serializable_results = {}
    
    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            'test_metrics': {
                'test_loss': float(model_results['test_metrics']['test_loss']),
                'author_accuracy': float(model_results['test_metrics']['author_accuracy']),
                'category_accuracy': float(model_results['test_metrics']['category_accuracy']),
                'author_predictions': [convert_to_serializable(x) for x in model_results['test_metrics']['author_predictions']],
                'author_true': [convert_to_serializable(x) for x in model_results['test_metrics']['author_true']],
                'category_predictions': [convert_to_serializable(x) for x in model_results['test_metrics']['category_predictions']],
                'category_true': [convert_to_serializable(x) for x in model_results['test_metrics']['category_true']]
            },
            'training_history': {
                'train_loss': [float(x) for x in model_results['training_history']['train_loss']],
                'val_loss': [float(x) for x in model_results['training_history']['val_loss']],
                'train_author_acc': [float(x) for x in model_results['training_history']['train_author_acc']],
                'val_author_acc': [float(x) for x in model_results['training_history']['val_author_acc']],
                'train_category_acc': [float(x) for x in model_results['training_history']['train_category_acc']],
                'val_category_acc': [float(x) for x in model_results['training_history']['val_category_acc']]
            }
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)

def main():
    # Đọc dữ liệu
    df = pd.read_csv('data/processed_data.csv')
    
    # Lọc top 5 tác giả và chủ đề
    top_5_authors = df['author'].value_counts().nlargest(5).index
    top_5_categories = df['category'].value_counts().nlargest(5).index
    df_filtered = df[df['author'].isin(top_5_authors) & df['category'].isin(top_5_categories)]
    
    # Label encoding
    author_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    df_filtered['author_encoded'] = author_encoder.fit_transform(df_filtered['author'])
    df_filtered['category_encoded'] = category_encoder.fit_transform(df_filtered['category'])
    
    # Chia dữ liệu
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    
    # Tạo datasets
    train_dataset = VietnameseNewsDataset(
        texts=train_df['processed_text'].values,
        author_labels=train_df['author_encoded'].values,
        category_labels=train_df['category_encoded'].values,
        tokenizer=tokenizer,
        remove_stopwords=False
    )
    
    val_dataset = VietnameseNewsDataset(
        texts=val_df['processed_text'].values,
        author_labels=val_df['author_encoded'].values,
        category_labels=val_df['category_encoded'].values,
        tokenizer=tokenizer,
        remove_stopwords=True
    )
    
    test_dataset = VietnameseNewsDataset(
        texts=test_df['processed_text'].values,
        author_labels=test_df['author_encoded'].values,
        category_labels=test_df['category_encoded'].values,
        tokenizer=tokenizer,
        remove_stopwords=True
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model configurations
    models = {
        'LSTM': LSTMMultiTask(
            vocab_size=len(tokenizer),
            embedding_dim=512,  # Tăng kích thước embedding
            hidden_dim=256,    # Tăng hidden size
            num_authors=len(top_5_authors),
            num_categories=len(top_5_categories),
            num_layers=3,      # Tăng số lớp
            dropout=0.4        # Tăng dropout
        ),
        'BiLSTM': BiLSTMMultiTask(
            vocab_size=len(tokenizer),
            embedding_dim=512,
            hidden_dim=256,
            num_authors=len(top_5_authors),
            num_categories=len(top_5_categories),
            num_layers=3,
            dropout=0.4
        ),
        'PhoBERT': PhoBERTMultiTask(
            num_authors=len(top_5_authors),
            num_categories=len(top_5_categories)
        )
    }
    
    # Training configuration
    num_epochs = 100
    criterion = DynamicMultiTaskLoss()
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=1e-4,              # Tăng learning rate
                                    weight_decay=0.01)    # Thêm weight decay
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        # Training
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            model_name=model_name
        )
        
        # Plot training history
        plot_training_history(history, model_name)
        
        # Load best model
        checkpoint = load_checkpoint(f'checkpoints/{model_name}_best.pt', model, device)
        if checkpoint is None:
            print(f"Using last model state for {model_name}")
        
        # Evaluate on test set
        test_results = evaluate_model(model, test_loader, criterion, device)
        results[model_name] = {
            'test_metrics': test_results,
            'training_history': history
        }
    
    # Save results
    save_results(results, 'results/multitask_results.json')
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    author_acc = [results[m]['test_metrics']['author_accuracy'] for m in model_names]
    category_acc = [results[m]['test_metrics']['category_accuracy'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, author_acc, width, label='Author Accuracy')
    plt.bar(x + width/2, category_acc, width, label='Category Accuracy')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png')
    plt.close()

if __name__ == "__main__":
    main()