import streamlit as st

import pickle 
import time

st.title(" Twitter Sentiment Analysis")

#load model 

model = pickle.load(open(r'C:\Users\91869\OneDrive\Desktop\streamlit\twitterSentiment\twitter_sentiment.pkl', 'rb'))

tweet = st.text_input("Enter your Tweet")

submit= st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict([tweet])
    end = time.time()
    st.write('Prediction time taken: ' , round(end-start , 2) ,'seconds')
    print(prediction[0])
    st.write("Prediction Sentiment is: " , prediction[0])
