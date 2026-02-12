# ACL Anthology Text Classification Project  
IFT6285 – Natural Language Processing (Université de Montréal)

---

## 📌 Project Overview

This project focuses on building and evaluating **text classification benchmarks** using metadata from the ACL Anthology.

The objectives were to:

- Construct custom NLP classification datasets via the ACL API  
- Compare multiple machine learning and deep learning approaches  
- Analyze model behavior across binary and multi-class tasks  
- Evaluate feature engineering strategies (BoW, TF-IDF, embeddings, transformers)

The project includes dataset construction, preprocessing, feature extraction, model comparison, performance evaluation, and analytical discussion.

---

## 📊 Research Tasks

### 🔹 Task 1 — Conference Prediction (Multi-class Classification)

**Goal:**  
Predict which conference a paper belongs to (ACL / EMNLP / LREC) using its **title + abstract**.

**Dataset:**

- Train: 6,000 samples (balanced)
- Test: 1,500 samples (balanced)
- 3 classes

This task evaluates topic-based text classification performance.

---

### 🔹 Task 2 — Article Length Prediction (Binary Classification)

**Goal:**  
Predict whether a paper is:

- **Long paper** (≥ 9 pages)  
- **Short paper** (≤ 8 pages)

based only on its **title + abstract**.

**Dataset:**

- Train: 12,800 samples
- Test: 3,189 samples
- Balanced binary classification

This task evaluates whether structural information (length) can be inferred from textual signals.

---

## 🛠️ Technical Stack

### Programming & Data Processing
- Python  
- Pandas  
- NumPy  

### Feature Engineering
- CountVectorizer  
- TfidfVectorizer  
- Word2Vec (Gensim)  
- DistilBERT (HuggingFace Transformers)  

### Models
- Logistic Regression  
- Support Vector Machine (SVC)  
- Multinomial Naive Bayes  
- Random Forest  
- Transformer embeddings + classical classifiers  

### Frameworks
- Scikit-learn  
- Gensim  
- HuggingFace Transformers  
- PyTorch  

---

## 🔬 Experiments & Model Comparison

A total of **14 systems** were evaluated across both tasks.

### Key Findings

- SVC consistently achieved strong performance.
- TF-IDF and Count-based features performed well for topic classification.
- Word2Vec underperformed compared to frequency-based approaches.
- DistilBERT embeddings improved performance in binary classification.
- Using bigrams significantly improved results over unigram baselines.

---

## 📈 Performance Overview

| Task | Best Model | Accuracy |
|------|------------|----------|
| Conference Prediction | CountVectorizer (1–2 gram) + SVC | ~0.63 |
| Article Length Prediction | DistilBERT + Logistic Regression | ~0.70 |

Additional evaluations were conducted on distributed benchmarks from the course.

---

## 🧠 Skills Demonstrated

- NLP dataset construction from API metadata  
- Text preprocessing & feature engineering  
- Classical ML vs Transformer-based modeling  
- Multi-class & binary classification  
- Model benchmarking and comparative analysis  
- Hyperparameter tuning (n-gram optimization)  
- Performance interpretation & error analysis  

---
## 🎓 Academic Context

Course: IFT6285 – Natural Language Processing  
Université de Montréal  
Fall 2024 