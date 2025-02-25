import csv
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

with open ("dataset.csv", encoding = 'utf8') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    texts = []
    emotions =[]
    for row in csv_reader:
        text = row.get('review')
        sentiment = row.get('sentiment')

        text = re.sub(r'[^A-Za-z ]','',text)

        texts.append(text)
        emotions.append(sentiment)

    vectorizer = CountVectorizer(binary = True, lowercase = True)
    vectorizer.fit(texts)
    x = vectorizer.transform(texts)

    x_train, x_val, y_train, y_val = train_test_split(x, emotions, train_size= 0.8)

    
    logistic_regretion = LogisticRegression(C=0.1, max_iter=900)
    logistic_regretion.fit(x_train, y_train)
    accuracy = accuracy_score(y_val, logistic_regretion.predict(x_val))

    pickle.dump(logistic_regretion, open('model.model', mode = 'wb'))
    pickle.dump(vectorizer, open('vectorizer.model', mode = 'wb'))
    
    

