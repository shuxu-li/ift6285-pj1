import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from data_processing import load_data, get_bert_embeddings
import gc



MAIN_FOLDER = "benchmarks"


def process_folder(folder_name):
    """traiter les données dans un dossier"""
    train_file = os.path.join(MAIN_FOLDER, folder_name, f"train-{folder_name}.csv")
    test_file = os.path.join(MAIN_FOLDER, folder_name, f"test-{folder_name}.csv")
    
    # loader les donées avec la fonction  
    train_texts, train_labels = load_data(train_file)
    test_texts, test_labels = load_data(test_file)
    
    # bert_svm
    print(f"=== Le modèle bert_svc en train de ebedding {folder_name} ...")
    train_features_bert = get_bert_embeddings(train_texts)
    test_features_bert = get_bert_embeddings(test_texts)

    model_bert_svc = SVC()
    print(f"=== Le modèle count_svc en train d'entrainer {folder_name} ...")
    model_bert_svc.fit(train_features_bert, train_labels)
    print(f"=== Le modèle count_svc en train de prédire {folder_name} ...")
    predict_bert_svm = model_bert_svc.predict(test_features_bert)
    prediction_bert = pd.DataFrame({'id': test_labels.index,'label': predict_bert_svm})
    prediction_bert.to_csv(f'sortie/{folder_name}-bert_svm.csv', index=False)
    print(f"distilbert_svm: \t {folder_name:<40} {accuracy_score(test_labels, predict_bert_svm)}" )


    # Count Vectorizer + SVM
    
    # model_count = Pipeline([("vectorizer2", CountVectorizer(ngram_range=(1, 2))), ("classifier", SVC())])
    # print(f"=============Le modèle count_svc en train de traite {folder_name} ...")
    # model_count.fit(train_texts, train_labels)
    # predict_count_svm = model_count.predict(test_texts)

    # prediction_count = pd.DataFrame({'id': test_labels.index,'label': predict_count_svm})
    # prediction_count.to_csv(f'{folder_name}/{folder_name}-count_svc.csv', index=False)

    # print(f"count_svm: \t {folder_name:<40}  acc: {accuracy_score(test_labels, predict_count_svm):3f}")
    # # relacher la mémoire des Cache
    
    gc.collect()

def main():

    for folder_name in os.listdir(MAIN_FOLDER):
        folder_path = os.path.join(MAIN_FOLDER, folder_name)
        if os.path.isdir(folder_path):  
            process_folder(folder_name)

if __name__ == "__main__":
    main()

