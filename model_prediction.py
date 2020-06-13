import sys
import numpy as np
import pickle
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
import warnings
from pymongo import MongoClient
import geograpy4
import argparse
warnings.filterwarnings('ignore')
from TweetProcessor import TweetProcessor as tp
tweet_processor = tp.TweetProcessor()
from UserModeling import UserModeling as um
userModeling = um.UserModeling()
import pandas as pd


def make_prediction(model_filepath, doc):
   # parse text
   if type(doc)==list:
      doc = ' '.join(doc)
   # process text 
   doc = ' '.join(tweet_processor.tweet_pipeline(doc))
   # load TF-IDF models
   text_tfidf = pickle.load(open("models/text_tfidf.model", 'rb'))
   emoji_tfidf = pickle.load(open("models/emoji_tfidf.model", 'rb'))
   
   # Create new tfidfVectorizers with old vocabularies
   text_tfidf_new = TfidfVectorizer(analyzer='word', 
                                min_df=5, 
                                ngram_range=(1, 3),
                                norm='l2', 
                                max_features=2000,
                                vocabulary = text_tfidf.vocabulary_)
   emoji_tfidf_new = TfidfVectorizer(analyzer='word', 
                                tokenizer=tweet_processor.tweet_preprocessing,
                                norm='l2',
                                lowercase=False,
                                vocabulary = emoji_tfidf.vocabulary_)
   # transform text to frequencies
   X_tfidf = text_tfidf_new.fit_transform([doc])
   X_emoji = emoji_tfidf_new.fit_transform([doc])   
   # combine features
   X_combined = np.hstack((X_tfidf.toarray(), X_emoji.toarray()))
   
   # load model from pickle
   model = pickle.load(open(model_filepath, 'rb'))
   result = model.predict(X_combined)
   prob = np.round(model.predict_proba(X_combined), 3)
   
   # format the prediction to output text
   if result.shape==(1,): # if sentiment model
      pred = result[0]
   elif np.shape(result)[1] == 11: # if emotion model
      # for emotion the probability prediction ,corresponds to labels:
      # ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'] 
      emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 
                  'sadness', 'surprise', 'trust']
      pred = [e for i,e in zip(result[0], emotions) if i>0]   
      if len(pred)==0:
         pred = [emotions[i] for i in np.argsort(prob)[0][:2]]
   else:
      pred = result
   if model_filepath == "models/finalized_sentiment.model":
      return {'sentiment': {'pred': pred, 'prob': prob[0].tolist()}}
   else:
      return {'emotion': {'pred': pred, 'prob': prob[0].tolist()}}

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--init", nargs='?',
                        const=True, default=False,
                        help="Activate init mode.")
   args = parser.parse_args()
   init = args.init
   with open("data/TextMiningInput.json", "r") as read_file:
      tweets = json.load(read_file)
   
   results = []
    
   file_names = "UserModeling/NamesList.csv"
   names_df = pd.read_csv(file_names)       
   list_names_df = userModeling.split_names(names_df)   
          
   for tweet in tweets['tweets']:   
      hate_speech = userModeling.predict_hate_speech(tweet['text'])
      gender = userModeling.extract_gender(tweet['username'],tweet['description'],list_names_df)
      age_group = userModeling.extract_age(tweet['username'],tweet['screen_name'])     
      location = geograpy4.get_place_context(tweet['user_location'],ignoreEstablishments=True)
      finalLocation = {}
      if len(location) > 0:
         finalLocation['lat'] = location[0]['lat']
         finalLocation['lon'] = location[0]['lon']
         finalLocation['location'] = location[0]['display_name']
      model_filepath, doc = "models/finalized_sentiment.model", tweet['text']
      sentimentResults = make_prediction(model_filepath, doc)
      model_filepath, doc = "models/finalized_emotion.model", tweet['text']
      emotionResults = make_prediction(model_filepath, doc)
      sentimentResults = json.loads(json.dumps(sentimentResults))
      emotionResults = json.loads(json.dumps(emotionResults))
      merged_dict = sentimentResults.copy()
      merged_dict.update(emotionResults)
      
      merged_dict.update(hate_speech)
      merged_dict.update(gender)
      merged_dict.update(age_group)
      if len(location) > 0:
         merged_dict.update(finalLocation)
      results.append(merged_dict)
   if init == True:
      dbclient = MongoClient('MongoURI')
      db = dbclient.test
      db.results.insert_many(results)

   print(results)
