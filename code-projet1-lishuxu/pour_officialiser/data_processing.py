import os
import pandas as pd
import numpy as np
import torch
import csv
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import CountVectorizer

# configurer pour profiter de GPU
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# instanciser tokenizer et model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# initialiser le countvectoriseur
vectorizer = CountVectorizer(ngram_range=(1, 2))

def load_data(file_path):
    """loader les donées de .csv et retirer les colones de features et de classes"""
    # 
    benchmarks_dir = os.path.join("benchmarks", file_path)

    file_name = os.path.basename(benchmarks_dir).replace('.csv', '')

    # la partie des features est le dernier élément du nom de fichier
    # ex: train-<name>-<task_name>-<features>.csv
    feature_part = file_name.split('-')[-1]
    
    data = pd.read_csv(file_path, sep=';', quotechar='"', quoting=csv.QUOTE_ALL, index_col='id')
    data = data.dropna(how='any')
    feature_columns = feature_part.split('+')

    for col in feature_columns:
        data[col] = data[col].astype(str)
    # prendre toute la part si on a juste un feature
    if len(feature_columns) == 1:
        texts = data[feature_columns[0]].values
    else:
        # combiner les colonnes de features
        texts = data[feature_columns].agg(' '.join, axis=1).values

    labels = data.iloc[:, -1]#.values   # la dernière colonne comme classe

    return texts, labels

def get_bert_embeddings(texts, batch_size=32):
    """générer embeddings de BERT"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size].tolist()
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# def count_vectorizer_features(texts, train=True):
#     """générer features de countvectorizer"""
#     if train:
#         return vectorizer.fit_transform(texts).toarray()
#     else:
#         return vectorizer.transform(texts).toarray()