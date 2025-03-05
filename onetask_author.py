#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from underthesea import word_tokenize
import json
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('data/processed_data.csv')

# Lấy top 5 tác giả có nhiều bài viết nhất
top_5_authors = df['author'].value_counts().nlargest(5).index
df_filtered = df[df['author'].isin(top_5_authors)]

def save_model_results(results_with_sw, results_without_sw, model_details):
    results = {
        "experiment_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_info": {
            "total_samples": len(df_filtered),
            "training_samples": len(X_train_with_sw),
            "test_samples": len(X_test_with_sw),
            "top_5_authors": df_filtered['author'].value_counts().nlargest(5).to_dict()
        },
        "models": {
            "with_stopwords": results_with_sw,
            "without_stopwords": results_without_sw,
            "detailed_metrics": model_details
        }
    }
    
    # Lưu kết quả vào file JSON
    with open('model_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def train_and_evaluate_models(X_train, X_test, y_train, y_test, vectorizer, case):
    # Vectorize text data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Khởi tạo các mô hình
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': LinearSVC(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    print(f"\nKết quả phân loại cho trường hợp {case}:")
    print("-" * 50)
    
    # Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        
        print(f"\nMô hình: {name}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

#%%
X_train_with_sw, X_test_with_sw, y_train, y_test = train_test_split(
    df_filtered['processed_text'],
    df_filtered['author'],
    test_size=0.2,
    random_state=42
)

X_train_without_sw, X_test_without_sw, _, _ = train_test_split(
    df_filtered['text_no_stopwords'],
    df_filtered['author'],
    test_size=0.2,
    random_state=42
)

tfidf = TfidfVectorizer()

# Huấn luyện và đánh giá mô hình cho cả 2 trường hợp
print("TOP 5 TÁC GIẢ CÓ NHIỀU BÀI VIẾT NHẤT:")
for author, count in df_filtered['author'].value_counts().nlargest(5).items():
    print(f"{author}: {count} bài viết")
# %%
train_and_evaluate_models(X_train_with_sw, X_test_with_sw, y_train, y_test, 
                         tfidf, "có stopwords")

train_and_evaluate_models(X_train_without_sw, X_test_without_sw, y_train, y_test,
                         tfidf, "không có stopwords")

#%%
import matplotlib.pyplot as plt

def plot_results(results_with_sw, results_without_sw):
    models = list(results_with_sw.keys())
    accuracy_with_sw = list(results_with_sw.values())
    accuracy_without_sw = list(results_without_sw.values())
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracy_with_sw, width, label='Với stopwords')
    rects2 = ax.bar(x + width/2, accuracy_without_sw, width, label='Không có stopwords')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('So sánh độ chính xác của các mô hình phân loại top 5 tác giả')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('onetask_author.png')


models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Lưu kết quả accuracy để vẽ biểu đồ
model_details = {
    "with_stopwords": {},
    "without_stopwords": {}
}

results_with_sw = {}
results_without_sw = {}
for name, model in models.items():
    # Với stopwords
    model.fit(tfidf.fit_transform(X_train_with_sw), y_train)
    y_pred_sw = model.predict(tfidf.transform(X_test_with_sw))
    results_with_sw[name] = accuracy_score(y_test, y_pred_sw)
    
    # Tính thêm precision, recall, f1-score
    precision_sw, recall_sw, f1_sw, _ = precision_recall_fscore_support(y_test, y_pred_sw, average='weighted')
    
    model_details["with_stopwords"][name] = {
        "accuracy": float(results_with_sw[name]),
        "precision": float(precision_sw),
        "recall": float(recall_sw),
        "f1_score": float(f1_sw),
        "classification_report": classification_report(y_test, y_pred_sw, output_dict=True)
    }
    
    # Không có stopwords
    model.fit(tfidf.fit_transform(X_train_without_sw), y_train)
    y_pred_no_sw = model.predict(tfidf.transform(X_test_without_sw))
    results_without_sw[name] = accuracy_score(y_test, y_pred_no_sw)
    
    precision_no_sw, recall_no_sw, f1_no_sw, _ = precision_recall_fscore_support(y_test, y_pred_no_sw, average='weighted')
    
    model_details["without_stopwords"][name] = {
        "accuracy": float(results_without_sw[name]),
        "precision": float(precision_no_sw),
        "recall": float(recall_no_sw),
        "f1_score": float(f1_no_sw),
        "classification_report": classification_report(y_test, y_pred_no_sw, output_dict=True)
    }

# Lưu kết quả
save_model_results(results_with_sw, results_without_sw, model_details)

# In tổng quan kết quả
print("\nTỔNG QUAN KẾT QUẢ:")
print("-" * 50)
print("\nKết quả với stopwords:")
for model_name, metrics in model_details["with_stopwords"].items():
    print(f"\n{model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")

print("\nKết quả không có stopwords:")
for model_name, metrics in model_details["without_stopwords"].items():
    print(f"\n{model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")

# Vẽ biểu đồ như cũ
plot_results(results_with_sw, results_without_sw)