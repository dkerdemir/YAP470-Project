#!/usr/bin/env python
# coding: utf-8

# Beginning of our project :)

# Reading our Dataset Below:

# In[12]:


#import sklearn.datasets
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import spacy
from pprint import pprint

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

import mlflow
from mlflow import log_metric, log_param, log_artifacts

from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[13]:


SEED = 12345
MB_dataset = pd.read_csv('mbti_1.csv')
print(MB_dataset)
MB_dataset.describe()
MB_dataset.isnull().sum()


# In[14]:


personalities = {'I':'Introversion', 'E':'Extroversion', 'N':'Intuition', 
        'S':'Sensing', 'T':'Thinking', 'F': 'Feeling', 
        'J':'Judging', 'P': 'Perceiving'}
MB_dataset.head()


# In[15]:


#make a bar chart of how often each personality type is mentioned in a post
MB_dataset["type"].value_counts()
colors = ['crimson', 'darkorchid', 'gray', 'orange', 'limegreen', 'blue', 'brown', 'hotpink', 'turquoise', 'cyan', 'salmon', 'cornflowerblue', 'olive', 'mediumslateblue', 'steelblue', 'gold']

MB_dataset["type"].value_counts().plot(kind="bar", color=['crimson', 'darkorchid', 'gray', 'orange', 'limegreen', 'blue', 'brown', 'hotpink', 'turquoise', 'cyan', 'salmon', 'cornflowerblue', 'olive', 'mediumslateblue', 'steelblue', 'gold' ])


# In[16]:


lemmatiser = WordNetLemmatizer()
useless_words = nltk.corpus.stopwords.words('english')


# In[17]:


def replace_symbols(text):
    text = re.sub('https?\S+', ' ', text) #Removing urls 
    text = re.sub("[^a-zA-Z]", " ", text) #Removing non-words
    text = re.sub(' +', ' ', text) #Removing consecutive whitespace   
    pers_types = ['INFP' ,'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP' ,'ISFP' ,'ENTJ', 'ISTJ','ENFJ', 'ISFJ' ,'ESTP', 'ESFP' ,'ESFJ' ,'ESTJ']
    pers_types = [p.lower() for p in pers_types]
    for t in pers_types:
        text = text.replace(t, "")
    p = re.compile("(" + "|".join(pers_types) + ")")
    text = " ".join([lemmatiser.lemmatize(w) for w in text.split(' ') if w not in useless_words])
    text = text.lower()
    return text

MB_dataset['cleaned_posts'] = MB_dataset['posts'].apply(replace_symbols)
MB_dataset.describe()
MB_dataset.head(10)


# In[18]:


print("\nPost before preprocessing:\n\n", MB_dataset['posts'][0])


# In[19]:


print("\nPost after preprocessing:\n\n", MB_dataset['cleaned_posts'][0])


# In[20]:


STOPWORDS.add('URL') # words to not consider
STOPWORDS.add('S')
labels = MB_dataset['type'].unique()
row, col = 4, 4
wc = WordCloud(stopwords=useless_words)

fig, ax = plt.subplots(4, 4, figsize=(20,15))

for i in range(4):
    for j in range(4):
        cur_type = labels[i*col+j]
        cur_ax = ax[i][j]
        df = MB_dataset[MB_dataset['type'] == cur_type]
        wordcloud = wc.generate(MB_dataset['cleaned_posts'].to_string())
        cur_ax.imshow(wordcloud)
        cur_ax.axis('off')
        cur_ax.set_title(cur_type)


# Feature Extraction

# In[21]:


sw = MB_dataset.copy()

sw['words_per_comment'] = sw['cleaned_posts'].apply(lambda x: len(x.split())/50)
sw.head()


# In[22]:


plt.figure(figsize=(15,10))
sns.swarmplot(data=sw, x="type", y="words_per_comment", palette = colors)
plt.show()


# In[23]:


sw['E/I'] = sw['type'].apply(lambda x: x[0] == 'E').astype('int')
sw['S/N'] = sw['type'].apply(lambda x: x[1] == 'S').astype('int')
sw['T/F'] = sw['type'].apply(lambda x: x[2] == 'T').astype('int')
sw['J/P'] = sw['type'].apply(lambda x: x[3] == 'J').astype('int')
sw.head()


# In[24]:


print(len(sw['posts'][1000]), len(sw['cleaned_posts'][1000]))


# In[25]:


nlp = spacy.load("en_core_web_sm")


# In[26]:


def tfidf_vectorize(max):
    TFIDF_vect = TfidfVectorizer(lowercase=True, stop_words='english', max_features=max)
    all_data_TFIDF = TFIDF_vect.fit_transform(sw['cleaned_posts'])
    return all_data_TFIDF


# In[27]:


def split_data(X,y,scaled,split_rate):
    if (scaled):
        scaler = StandardScaler(with_mean = False)
        X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_rate, random_state=42, stratify=y)
    return X_train, y_train, X_test, y_test


# In[28]:


def logistic_regression(X_train,y_train,X_test,y_test,K, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        logistic_regression_classifier = LogisticRegression()
        logistic_regression_classifier.fit(X_train, y_train)
        prediction_test = logistic_regression_classifier.predict(X_test)
        print('Logistic_regression:')
        if(K > 0):
            skf = StratifiedKFold(n_splits=K)
            print('Cross Validation Accuracy:', round(cross_val_score(logistic_regression_classifier, X_train, y_train, cv=skf).mean(), 3))
        else:
            prediction_train = logistic_regression_classifier.predict(X_train)
            print('Train Accuracy:', round(accuracy_score(y_train, prediction_train), 3))
            
        accuracy = balanced_accuracy_score(y_test, prediction_test)
        f1Score = f1_score(y_test, prediction_test, average='weighted')
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1_score", f1Score)
        mlflow.sklearn.log_model(logistic_regression_classifier, "Logistic Regression")
        print('Test Accuracy:', round(accuracy, 3), ' - F1:', round(f1Score, 3))        
        


# In[29]:


def SVM(X_train,y_train,X_test,y_test,K, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model_svc=SVC(kernel="linear")
        model_svc.fit(X_train,y_train)
        prediction_test = model_svc.predict(X_test)
        print('Support Vector Machine:')
        if(K > 0):
            skf = StratifiedKFold(n_splits=K)
            print('Cross Validation Accuracy:', round(cross_val_score(model_svc, X_train, y_train, cv=skf).mean(), 3))
        else:
            prediction_train = model_svc.predict(X_train)
            print('Train Accuracy:', round(accuracy_score(y_train, prediction_train), 3))
            
        accuracy = balanced_accuracy_score(y_test, prediction_test)
        f1Score = f1_score(y_test, prediction_test, average='weighted')
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1_score", f1Score)
        mlflow.sklearn.log_model(model_svc, "Support Vector Machine")
        print('Test Accuracy:', round(accuracy, 3), ' - F1:', round(f1Score, 3)) 


# In[30]:


def naive_bayes(X_train,y_train,X_test,y_test,K, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model_multinomial_nb=MultinomialNB()
        model_multinomial_nb.fit(X_train,y_train)
        prediction_test = model_multinomial_nb.predict(X_test)
        print('Naive Bayes:')
        if(K > 0):
            skf = StratifiedKFold(n_splits=K)
            print('Cross Validation Accuracy:', round(cross_val_score(model_multinomial_nb, X_train, y_train, cv=skf).mean(), 3))
        else:
            prediction_train = model_multinomial_nb.predict(X_train)
            print('Train Accuracy:', round(accuracy_score(y_train, prediction_train), 3))
            
        accuracy = balanced_accuracy_score(y_test, prediction_test)
        f1Score = f1_score(y_test, prediction_test, average='weighted')
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1_score", f1Score)
        mlflow.sklearn.log_model(model_multinomial_nb, "Naive Bayes")
        print('Test Accuracy:', round(accuracy, 3), ' - F1:', round(f1Score, 3)) 


# In[31]:


def random_forest(X_train,y_train,X_test,y_test,K, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model_forest=RandomForestClassifier(max_depth=10)
        model_forest.fit(X_train,y_train)
        prediction_test = model_forest.predict(X_test)
        print('Random Forest:')
        if(K > 0):
            skf = StratifiedKFold(n_splits=K)
            print('Cross Validation Accuracy:', round(cross_val_score(model_forest, X_train, y_train, cv=skf).mean(), 3))
        else:
            prediction_train = model_forest.predict(X_train)
            print('Train Accuracy:', round(accuracy_score(y_train, prediction_train), 3))
            
        accuracy = balanced_accuracy_score(y_test, prediction_test)
        f1Score = f1_score(y_test, prediction_test, average='weighted')
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1_score", f1Score)
        mlflow.sklearn.log_model(RandomForestClassifier, "Random Forest")
        print('Test Accuracy:', round(accuracy, 3), ' - F1:', round(f1Score, 3)) 


# In[32]:


def xgb(X_train,y_train,X_test,y_test,K, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model_xgb=XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.1)
        model_xgb.fit(X_train,y_train)
        prediction_test = model_xgb.predict(X_test)
        print('XGBoost:')
        if(K > 0):
            skf = StratifiedKFold(n_splits=K)
            print('Cross Validation Accuracy:', round(cross_val_score(model_xgb, X_train, y_train, cv=skf).mean(), 3))
        else:
            prediction_train = model_xgb.predict(X_train)
            print('Train Accuracy:', round(accuracy_score(y_train, prediction_train), 3))
            
        accuracy = balanced_accuracy_score(y_test, prediction_test)
        f1Score = f1_score(y_test, prediction_test, average='weighted')
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1_score", f1Score)
        mlflow.sklearn.log_model(XGBClassifier, "XGBoost")
        print('Test Accuracy:', round(accuracy, 3), ' - F1:', round(f1Score, 3)) 


# In[33]:


def catBoost(X_train,y_train,X_test,y_test,K, params):
    with mlflow.start_run():
        mlflow.log_params(params)
        model_cat=CatBoostClassifier(iterations=100, depth = 3, loss_function='MultiClass',eval_metric='MultiClass',task_type='GPU',verbose=False)
        model_cat.fit(X_train,y_train)
        prediction_test = model_cat.predict(X_test)
        print('CatBoost:')
        if(K > 0):
            skf = StratifiedKFold(n_splits=K)
            print('Cross Validation Accuracy:', round(cross_val_score(model_cat, X_train, y_train, cv=skf).mean(), 3))
        else:
            prediction_train = model_cat.predict(X_train)
            print('Train Accuracy:', round(accuracy_score(y_train, prediction_train), 3))
            
        accuracy = balanced_accuracy_score(y_test, prediction_test)
        f1Score = f1_score(y_test, prediction_test, average='weighted')
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1_score", f1Score)
        mlflow.sklearn.log_model(CatBoostClassifier, "catBoost")
        print('Test Accuracy:', round(accuracy, 3), ' - F1:', round(f1Score, 3))


# In[37]:


def run_models(split_rate, scaled, K, maxFeature):
    X_vec = tfidf_vectorize(100000)
    y = sw[['type']]
    target_encoder=LabelEncoder()
    y=target_encoder.fit_transform(y)
    X = SelectKBest(chi2, k=maxFeature).fit_transform(X_vec, y)
    
    params = {
        "split_rate": split_rate,
        "scaled": scaled,
        "SCV_folds": K ,
        "FeatureNum": maxFeature
    }
    
    X_train,y_train,X_test,y_test = split_data(X,y,scaled,split_rate)
    logistic_regression(X_train,y_train,X_test,y_test, K, params)
    SVM(X_train,y_train,X_test,y_test,K, params)
    naive_bayes(X_train,y_train,X_test,y_test,K, params)
    random_forest(X_train,y_train,X_test,y_test,K, params)
    xgb(X_train,y_train,X_test,y_test,K, params)
    catBoost(X_train,y_train,X_test,y_test,K, params)


# In[38]:


def main():
    run_models(0.2, 'true', 5, 1000)
    


# In[39]:


main()


# In[42]:


get_ipython().system('mlflow ui')


# 
