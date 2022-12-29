import json
import pandas as pd
import datetime
import collections
import collections.abc
from collections.abc import Mapping
from tweetkit.auth import BearerTokenAuth
from tweetkit.client import TwitterClient
collections.Mapping = collections.abc.Mapping
from tweetkit.models.request import TwitterRequestScheduler
import psycopg2
from sqlalchemy import create_engine
from collections.abc import Sequence
collections.Sequence = collections.abc.Sequence
from preprocess import mainPreProcess

import logging
log = logging.getLogger(__name__)

def QuerySetUp(other_conditions, start_time, end_time):
   #authentication values
  bearer_token  = 'AAAAAAAAAAAAAAAAAAAAACu%2BRwEAAAAAV70y9V%2FsDf8ashVkd3EJP%2F5p8UU%3DAW8QYLVJQy3hnODINrp1gLZlAeuY2yjFAk41CitAKmlVdLnuNq'
  auth = BearerTokenAuth()
  auth.bearer_token = bearer_token
  client = TwitterClient(auth=auth)

  #initialize query, start/end times, and tweetlist

  tweet_list = []

## ISO-8601 format
  # start_time_ = start_time
  # end_time_ = end_time
  start_time_ = start_time 
  end_time_ = end_time

# build a query e.g.'(#Ian OR "Hurricane Ian" OR #HurricaneIan) -is:retweet has:geo lang:en',
  keywords = ['#foodpoisoning', '#stomachache', '"food poison"', '"food poisoning"', 'stomachache', 'vomit', 'puke', 'diarrhea', '"the runs"', 'nausea', '"stomach cramp"', 'nauseous']
  search_keywords = ' OR '.join(keywords) 
  final_query = '('+search_keywords+') '+ other_conditions

# set other parameters
  tweet_fields_ = [ 'author_id', 'created_at', 'geo', 'id', 'lang', 'text'] 
  user_fields_ = ['id', 'location', 'name', 'username', 'verified'] 
  place_fields_ = ['country', 'country_code','geo', 'id', 'name', 'place_type']

#unused fields to possibly use later:
#expansions_ =['author_id', 'geo.place_id']
#media_fields_= ['alt_text', 'media_key', 'preview_image_url','public_metrics', 'type', 'url']
#poll_fields_ = ['duration_minutes', 'end_datetime', 'id', 'options', 'voting_status']
#"""See more details about the params in tweets_fullarchive_search here: https://tweetkit.readthedocs.io/en/latest/tweetkit.requests.html#tweetkit.requests.tweets.Tweets.tweets_fullarchive_search"""


  paginator = client.tweets.tweets_fullarchive_search(
    query = final_query,
    start_time = start_time_,
    end_time = end_time_,
    max_results = 500,
    paginate = True,
    tweet_fields = tweet_fields_ ,
    #expansions = expansions_ ,
    #media_fields = media_fields_ ,
    #poll_fields = poll_fields_ ,
    user_fields = user_fields_ ,
  )



  start_time = datetime.datetime.strptime(start_time_, '%Y-%m-%dT%H:%M:%SZ')
  end_time = datetime.datetime.strptime(end_time_, '%Y-%m-%dT%H:%M:%SZ')
  percentage_remaining = 0.0

  total_period = (end_time - start_time).total_seconds()


  for tweet in paginator.content:
    created_at = datetime.datetime.strptime(tweet['data']['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ')
    # tweets.append(tweet)
    # remaining_period = (end_time - created_at).total_seconds()
    # percentage_remaining = round(remaining_period * 100 / total_period, 2)
    # print('\rTweet Count: {:3.0f}%, {}'.format(percentage_remaining, len(tweets)), end='')

    if 'geo' not in tweet['data']:
      tweet_tuple = {'id' : tweet['data']['id'],
                  'author_id': tweet['data']['author_id'],
                  'tweet_text': tweet['data']['text'],
                  'created_at': tweet['data']['created_at'],
                  'geo': 'NULL',
                  }
    else:
        tweet_tuple = {'id' : tweet['data']['id'],
                  'author_id': tweet['data']['author_id'],
                  'tweet_text': tweet['data']['text'],
                  'created_at': tweet['data']['created_at'],
                  'geo': tweet['data']['geo']['place_id']}
    
    tweet_list.append(tweet_tuple)
    
  # if percentage_remaining != 100.00:
  log.info('\r {}-{} Collected Tweet Count:  {}'.format(start_time, end_time,len(tweet_list)))
  return tweet_list

def toDataFrame(tweets):
  tweet_df = pd.DataFrame()
  tweet_df = pd.DataFrame.from_records(tweets)
  return tweet_df

def ConnectToDatabase(tweet_df, table):
  conn_string = 'postgresql://dbadmin:8x6Hh!Jsr#tMGh$G@usda-foodpoisoning.wpi.edu:5432/MQP22'
  db = create_engine(conn_string)
  conn = db.connect()

  tweet_df.to_sql(table, con=conn, if_exists='replace', index=False)
  
  conn = psycopg2.connect(conn_string)
  conn.autocommit = True
  cursor = conn.cursor()

  #for debugging purposes
  select = '''select * from test_tweets'''
  cursor.execute(select)
  for i in cursor.fetchall():
    print(i)

  conn.close()


def mainCollect(start_time, end_time, hasgeo=True):
  #Set to false to include tweets that may not include geo-location

  #set search conditions
  no_geo_conditions= '-is:retweet lang:en -has:geo'
  geo_conditions= '-is:retweet lang:en has:geo' # place_country:US


  if hasgeo:
    cond = geo_conditions
  
  else:
    cond = no_geo_conditions
  
  tweets = QuerySetUp(cond,start_time, end_time)  
  tweet_df = toDataFrame(tweets)
  processed_tweet_df = mainPreProcess(tweet_df)
  # ConnectToDatabase(tweet_df, 'temp_tweets')
  return processed_tweet_df

if __name__ == "__main__":
  start_time= '2020-05-25T00:00:01Z' 
  end_time= '2020-05-25T0:02:00Z'
  mainCollect(start_time, end_time)