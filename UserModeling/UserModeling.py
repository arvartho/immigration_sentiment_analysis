import datetime
import re
import math
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
'''
import spacy
nlp = spacy.load('en')
'''
# Download the set of stop words the first time
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import joblib

class UserModeling():
   def __init__(self, handle_negation=False):
      self.male_nouns = ["actor", "author", "boy", "brother", "dad", "daddy", "man", "father"
         , "grandfather", "husband", "king", "man", "sir", "son", "uncle", "wizard"
         , "waiter", "guy"]
      self.female_nouns = ["actress", "authoress", "girl", "bride", "sister", "mum", "mummy", "woman", "mother",
                      "goddess", "grandmother", "grandmom", "wife", "queen", "woman", "madam", "daughter", "aunt",
                      "witch", "waitress"]
      self.hate_speech_model = joblib.load('UserModeling/hate_speech_model.pkl')
      self.tfidf_vec = joblib.load('UserModeling/tfidf_vectorizer.pkl')

      self.stop_words = set(stopwords.words('english'))  # Set improves performance
      self.unwatedChars = '''()-[]{};:'"\,<>./@#$%^&*_~1234567890'''

   def extract_age_group(self,age):
      if (not math.isnan(float(age))):  # checking for nulls. Weird way but true
         age = int(age)
         if (age <= 30):
            age_group = "Young"
         elif (age > 30 and age <= 60):
            age_group = "Middle_aged"
         else:
            age_group = "Elder"
      else:
         age_group = 'nan'
      return age_group

   def extract_age(self,name, username):
      name_and_username = name + " " + username
      now = datetime.datetime(2020, 5, 17)
      age = 'nan'

      try:
         if (re.search("(?<!\d)(\d{2}|\d{4})(?!\d)(?!%)", name_and_username)):
            birth_date = int(re.findall("(?<!\d)(\d{2}|\d{4})(?!\d)", name_and_username)[0])
            if (birth_date > 1950 and birth_date < 2000):
               age = now.year - birth_date
            if (birth_date > 50 and birth_date < 100):
               age = now.year - int('19' + str(birth_date))
      except:
         return {"age_group": self.extract_age_group(age)}
      return {"age_group": self.extract_age_group(age)}

   def index_for_letter(self,username):

      letter = username.lower()[0]
      idx = 'nan'

      if letter == 'a':
         idx = 0
      elif letter == 'b':
         idx = 1
      elif letter == 'c':
         idx = 2
      elif letter == 'd':
         idx = 3
      elif letter == 'e':
         idx = 4
      elif letter == 'f':
         idx = 5
      elif letter == 'g':
         idx = 6
      elif letter == 'h':
         idx = 7
      elif letter == 'i':
         idx = 8
      elif letter == 'j':
         idx = 9
      elif letter == 'k':
         idx = 10
      elif letter == 'l':
         idx = 11
      elif letter == 'm':
         idx = 12
      elif letter == 'n':
         idx = 13
      elif letter == 'o':
         idx = 14
      elif letter == 'p':
         idx = 15
      elif letter == 'q':
         idx = 16
      elif letter == 'r':
         idx = 17
      elif letter == 's':
         idx = 18
      elif letter == 't':
         idx = 19
      elif letter == 'u':
         idx = 20
      elif letter == 'v':
         idx = 21
      elif letter == 'w':
         idx = 22
      elif letter == 'x':
         idx = 23
      elif letter == 'y':
         idx = 24
      elif letter == 'z':
         idx = 25

      return idx

   def extract_gender_from_username(self,username, list_names_df):
      gender = 'nan'

      idx = self.index_for_letter(username)
      if idx == 'nan':
         return 'nan'

      names_df = list_names_df[self.index_for_letter(username)]

      for name_idx, name_row in names_df.iterrows():
         if (re.search(rf'\b({name_row["name"]})\b', username, re.IGNORECASE)):
            if (name_row['gender'] == 'male'):
               gender = "Male"
               break
            elif (name_row['gender'] == 'female'):
               gender = "Female"
               break
      return gender

   def extract_gender_from_decription(self,description):

      gender = 'nan'
      male_nouns_n = 0
      female_nouns_n = 0
      for noun in self.male_nouns:
         if (re.search(rf'\b({noun})\b', description, re.IGNORECASE)):
            male_nouns_n += 1
      for noun in self.female_nouns:
         if (re.search(rf'\b({noun})\b', description, re.IGNORECASE)):
            female_nouns_n += 1
      if (male_nouns_n > female_nouns_n):
         gender = "Male"
      if (female_nouns_n > male_nouns_n):
         gender = "Female"
      return gender

   def extract_gender(self,username, description, list_names_df):
      gender = self.extract_gender_from_username(username, list_names_df)
      if gender == 'nan':
         gender = self.extract_gender_from_decription(description)
      return {"gender": gender}

   def split_names(self,names_df):
      list_names_df = []
      str_idx = 0
      for i, row in names_df.iterrows():
         if (i + 1) == names_df.shape[0]:
            list_names_df.append(names_df.iloc[str_idx: i + 1])
            break
         if row['letter'] != names_df.iloc[i + 1, 0]:
            list_names_df.append(names_df.iloc[str_idx: i + 1])
            str_idx = i + 1

      return list_names_df

   def split_names(self,names_df):
      list_names_df = []
      str_idx = 0
      for i, row in names_df.iterrows():
         if (i + 1) == names_df.shape[0]:
            list_names_df.append(names_df.iloc[str_idx: i + 1])
            break
         if row['letter'] != names_df.iloc[i + 1, 0]:
            list_names_df.append(names_df.iloc[str_idx: i + 1])
            str_idx = i + 1

      return list_names_df

   def predict_hate_speech(self, tweet):

      tweet = re.sub('@\S*', '', tweet)
      tweet = re.sub('http\S+', '', tweet)
      tweet = tweet.lower()

      word_tokens = word_tokenize(tweet)
      filtered_tweet = [word for word in word_tokens if not word in self.stop_words]
      tweet = ' '.join(word for word in filtered_tweet)

      for punc in list(self.unwatedChars):
         tweet = tweet.replace(punc, '')
      '''
      for token in nlp(tweet):
         lemmatizedTweet.append(token.lemma_)
      tweet = ' '.join(word for word in lemmatizedTweet)
      '''
      vectorized_tweet = self.tfidf_vec.transform([tweet])

      pred = self.hate_speech_model.predict(vectorized_tweet)
      return {"hate_speech": pred[0]}