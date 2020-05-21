import sys
import numpy as np
import pickle
import random


def make_prediction(model_filepath, x):
   # load model from pickle
   model = pickle.load(open(model_filepath, 'rb'))
   result = model.predict(x.reshape(1, -1))
   prob = np.round(model.predict_proba(x.reshape(1, -1)), 3)
   # format the prediction to output text
   if result.shape==(1,):
      pred = result[0]
   elif np.shape(result)[1] == 11:
      # for emotion the probability prediction ,corresponds to labels:
      # ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'] 
      emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 
                  'sadness', 'surprise', 'trust']
      pred = [e for i,e in zip(result[0], emotions) if i>0]   
   else:
      pred = result
   return pred, prob 

if __name__ == "__main__":
   if len(sys.argv)==3:
      model_filepath, x = sys.argv[1:]

      print(make_prediction(model_filepath, np.asarray(x)))
   else:
      raise Exception("Wrong number of arguments: Function make_prediction, requires 2 arguments %d were given ARGS: model_filepath, x" % len(sys.argv))