import pandas as pd
import json
import argparse
import re, os, time
import glob
from sqlalchemy.engine import create_engine



def normalize(token):
  lowercased_token = token.lower()
  if token.startswith("@") and len(token)>1:
      return "@USER"
  elif len(token) >=3:
      if token[1] == "@":
          return token[0]+"@USER"
  if lowercased_token.startswith("http") or lowercased_token.startswith("www"):
      return "HTTPURL"
  for i in ['.com', '.edu', '.gov']:
      if i in lowercased_token:
          return "HTTPURL"
  else:
      return token

def create_tokens(text):
  text = text.replace('\n',' ').replace('\r',' ').replace("’", "'").replace("…", "...")
  tokens = text.split(' ')
  tokens = [normalize(token) for token in tokens]
  new_text = ' '.join(tokens)
  #temp = list(filter(None, re.split('([,.!?:()[\]"\s+])', new_text)))
  #tweet_split = list(filter(str.strip, temp))
  #fulltext = " ".join 
  return new_text

def mainPreProcess(prep_tweets):
    # conn_string = 'postgresql://dbadmin:8x6Hh!Jsr#tMGh$G@usda-foodpoisoning.wpi.edu:5432/MQP22'
    # db = create_engine(conn_string)
    # retrieve_conn = db.connect()
    # query = pd.read_sql("select * from \"test_tweets\"", db)
    # prep_tweets = pd.DataFrame(query, columns = ['id','author_id','tweet_text','created_at','geo'])
    prep_tweets['tweet_token'] = prep_tweets['tweet_text'].map(lambda text: create_tokens(text))
    # prep_tweets.to_csv('prep_tweets.csv')
    return prep_tweets

# if __name__ == "__main__":
#   mainPreProcess()

