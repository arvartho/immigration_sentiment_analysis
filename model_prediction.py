import sys
import numpy as np
import pickle
import random
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
import warnings
warnings.filterwarnings('ignore')
from TweetProcessor import TweetProcessor as tp
tweet_processor = tp.TweetProcessor()

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
   return {'pred': pred, 'prob': prob[0].tolist()} 

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--model', dest = "model", help="Path for sentiment/emotion model. Models can be found under \"models\" directory",
                    type=str)
   parser.add_argument('--text', dest="doc", help="Text for prediction",
                    type=str)
   args = parser.parse_args()
   
   model_filepath, doc = args.model, args.doc
   result = make_prediction(model_filepath, doc)
   print(result)
