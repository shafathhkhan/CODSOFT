import pandas as pd
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,classification_report

train_file = "data\\train_data.txt"
test_file="data\\test_data.txt"

with open(train_file, 'r',encoding='utf-8') as f:
    train_lines=f.readlines()

def parse_train_line(line):
    parts=line.strip().split(" ::: ")   
    return parts[0],parts[1],parts[2],parts[3]
train_data=[parse_train_line(line) for line in train_lines]
df_train= pd.DataFrame(train_data,columns=["ID","TITLE","GENRE","DESCRIPTION"])

with open(test_file,'r', encoding='utf-8') as f:
    test_lines = f.readlines()

def parse_test_line(line):
    parts=line.strip().split(" ::: ")
    return parts[0],parts[1],parts[2]
test_data=[parse_test_line(line) for line in test_lines]
df_test= pd.DataFrame(test_data,columns=["ID","TITLE","DESCRIPTION"])

print("Train Data loaded successfully")
print(df_train.head())
print("Test Data loaded successfully")
print(df_test.head())

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^A-Za-z0-9\s]','',text)
    return text

df_train['cleaned_desc']= df_train['DESCRIPTION'].astype(str).apply(clean_text)
df_test['cleaned_desc']=df_test['DESCRIPTION'].astype(str).apply(clean_text)

vectorizer=TfidfVectorizer(max_features=5000)
x_train = vectorizer.fit_transform(df_train['cleaned_desc'])
y_train=df_train['GENRE']
x_test = vectorizer.transform(df_test['cleaned_desc'])

classifier = LogisticRegression(max_iter=500)
classifier.fit(x_train,y_train)

df_test['PREDICTED_GENRE'] = classifier.predict(x_test)

output_file = "predicted_genre.txt"
df_test[['ID','TITLE','PREDICTED_GENRE']].to_csv(output_file,sep='\t' ,index=False,header= False)
print(f"The predictions saved to {output_file}")
