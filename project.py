import warnings
warnings.filterwarnings("ignore")

import pandas as pd

df= pd.read_csv('twitter_sentiment.csv' , header=None , index_col=0)
df = df[[2,3]].reset_index(drop=True)#df.columns = ['sentiment' , 'text']

df.columns=['sentiment' , 'text']
df.head()

df.isnull().sum()
df.dropna(inplace=True)
df['text'].apply(len).value_counts()
df.sample(10)

##DATA PREPROCESSING 
#do as you want 

##DATA CLEANING

df['text'] = df['text'].apply(lambda x : x.lower())


#WORD VISUALIZATION 

from wordcloud import WordCloud , STOPWORDS
stopwords = set(STOPWORDS)
stopwords_list = list(stopwords)


#train test split 

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(df['text'] , df['sentiment'] , test_size = 0.2 , random_state =0)

#MODEL BUILDING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

clf = Pipeline( [ ('tfid' , TfidfVectorizer(stop_words=stopwords_list)) , ('rfc' , RandomForestClassifier(n_jobs=-1))])

clf.fit(X_train , Y_train)

import pickle
pickle.dump(clf , open("twitter_sentiment.pkl" ,'wb'))

#EVALUATION

#from sklearn.metrics import classification_report
#y_pred = clf.predict(X_test)
#print(classification_report(Y_test , y_pred))

