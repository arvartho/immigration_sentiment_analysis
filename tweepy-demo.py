import tweepy
import json
import csv
import sys
import os.path
import datetime as dt
import time

api_file = sys.argv[1]
with open(api_file, 'r') as f:
   api_key_dict = json.load(f)

consumer_key = api_key_dict['consumer_key']
consumer_secret = api_key_dict['consumer_secret']
access_token = api_key_dict['access_token']
access_token_secret = api_key_dict['access_token_secret']

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth) 

# Creating the API object while passing in auth information
api = tweepy.API(auth)

# The search term you want to find
queries = ["immigration", "migrant", "invasion", "immigration to Europe", "refugee",
      "#GreeceUnderAttack", "#IStandWithGreece", "#GreeceDefendsEurope", '#StopIslam', 
      '#refugeecrisis', 'asylum seeker', '#asylumseeker', '#asylum', '#RefugeesWelcome',
      '#RefugeesUnderAttack'  ]
# Language code (follows ISO 639-1 standards)
language = "en"
# Output file
# tweet_store = 'tweet_immigration_db.csv'
tweet_store = sys.argv[2]
# The number of tweets to return per page, up to a max of 100.
rpp = 50
# The page number (starting at 1) to return, up to a max of roughly 1500 results (based on rpp * page.
page = 10
#timestamp
ts = time.time()
timestamp = dt.datetime.fromtimestamp(ts).strftime('%Y/%m/%d %H:%M:%S')

mode = 'a' if os.path.isfile(tweet_store) else 'w'
with open(tweet_store, mode) as csvfile:
   header = ['keyword', 'timestamp', 'username', 'screen_name', 'user_location', 'user_description', 
         'user_followers_count', 'user_friends_count', 'user_favourites_count', 'text', 'place', 
         'coordinates', 'favorite_count', 'hashtags', 'retweet_count']
   writer = csv.DictWriter(csvfile, fieldnames=header)
   writer.writeheader()

for q in queries:
   print("Capturing keyword: %s" % q)
   results = api.search(q=q, lang=language, rpp=rpp)

   # foreach hashtag through all tweets pulled
   for tweet in results:
      if type(tweet) != str:         
         csvfile = open(tweet_store, 'a')
         writer = csv.writer(csvfile)
         # Captured information:
         # keyword, timestamp, username, screen_name, user_location, user_description, 
         # user_followers_count, user_friends_count, user_favourites_count, text, place, coordinates, 
         # favorite_count, hashtags, retweet_count
         keyword = q         
         username = tweet.user.name
         screen_name = tweet.user.screen_name
         user_location = tweet.user.location
         user_description = tweet.user.description
         user_followers_count = tweet.user.followers_count
         user_friends_count = tweet.user.friends_count
         user_favourites_count = tweet.user.favourites_count         
         text = tweet.text.strip()
         place = tweet.place
         coordinates = tweet.coordinates
         favorite_count = tweet.favorite_count
         hashtags = [h['text'] for h in tweet.entities['hashtags']]
         retweet_count = tweet.retweet_count
         writer.writerow([keyword, timestamp, username, screen_name, user_location, user_description, 
               user_followers_count, user_friends_count, user_favourites_count, text, place, 
               coordinates, favorite_count, hashtags, retweet_count])
         csvfile.close()
      else:
         print("Keyword %s not found" % q)


