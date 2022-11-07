## First install the package 
using pip or git clone the source code.
```
pip install tweetkit
```
(Source code of [tweetkit](https://github.com/ysenarath/tweetkit))

---
## Full archive search

1. Sample code: full_archive_search_sample_code.ipynb

   You need to input the ***bearer token***
2. Learn the ***rate limit*** and query parameters [here](https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all).

3. Learn how to build a query [here](https://developer.twitter.com/en/docs/twitter-api/tweets/counts/integrate/build-a-query).

### Tasks

1. Test time range: set different time range to see whether we can retrieve all the tweets.

    e.g. split 2 mins into 12\*10s, investigate if the number of collected tweets with one run in 2 mins equals to the sum of 12 runs.
      
    Run experiments for different date and different time (morning, afternoon, night) 

2. Add conditon to collect tweets in the US. 

    Report the difference between the # of tweets collected with and without geolocation filtering conditions.

3. Look into the entities in the tweet.fields.
https://developer.twitter.com/en/docs/labs/annotations

   Extract the corresponding entity tokens from each tweet to investigate whether the annotations provided by Twitter is accurate. 
   
---
## Stream search:
1. Sample code: filtered_stream_sample_code.ipynb
   You need to input the ***bearer token***
2. Learn the ***rate limit*** and parameters [here](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/api-reference/post-tweets-search-stream-rules).
2. Learn how to build rules [here](https://developer.twitter.com/en/docs/twitter-api/tweets/filtered-stream/integrate/build-a-rule). The rule is similar to the query in full_archive_search.

### Tasks

1. Using the same rules and query to see whether "full archive search" and "filtered stream" return the same number of tweets.
2. Test the stability of the script. (Can it run for 12hr/24hr?)

## :point_right: Filter out tweets from News agencies (or government):newspaper:
