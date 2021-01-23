#importing important modules that we will need
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("news.csv")
df.head()
#store feture in variable 
text=df['text']
#store label we will predict in variable 
labels=df['label']
labels.head()
# split our data to train data and test data 
X_train,x_test,y_train,y_test=train_test_split(text,labels,test_size=0.2,random_state=7)

#make object from tfidvectotizer type words english max words 7
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# train the module on the training data 
tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


#Initialize a PassiveAggressiveClassifier
pac=RandomForestClassifier()
#trian y dataset

pac.fit(tfidf_train,y_train)
# what we will pridect 
y_pred=pac.predict(tfidf_test)

# compare our pridections to the testset
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
# print the accuracy 100% 
# 
# 
# lets pridict new information after we train the moudel 
input_data = [input()]
vectorized_input_data = tfidf_vectorizer.transform(input_data)
prediction = pac.predict(vectorized_input_data)
print(prediction)



