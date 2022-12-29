import numpy as np
import torch
import os
import time
import datetime
from multiprocessing import Process, cpu_count, Queue
from tweet_collectiion import mainCollect
from predict import mainPredict
from send_email import send_eamil
import logging


def producer(queue, start_time_iso, end_time_iso, hasgeo):
 # the input start_time_iso and end_time_iso are UTC time in ISO-8601 format
    st=start_time_iso
    start_time_ = datetime.datetime.strptime(start_time_iso, '%Y-%m-%dT%H:%M:%SZ')
    end_time = datetime.datetime.strptime(end_time_iso, '%Y-%m-%dT%H:%M:%SZ')
    time_interval = 60*24   # 24 Hours, in minutes

    while start_time_< end_time:
      # print(start_time_,end_time)
      end_time_ = start_time_ + datetime.timedelta(hours=0,minutes = time_interval)
      end_time_iso_ = datetime.datetime.strftime(end_time_,'%Y-%m-%dT%H:%M:%SZ')
      if end_time_ >= end_time:
          end_time_ = end_time
          end_time_iso_ = end_time_iso
      try:    
      ##########
        processed_tweet_df = mainCollect(start_time_iso, end_time_iso_, hasgeo=hasgeo)
        
        ## send into queue (pass to consumer)
        # queue.put(processed_tweet_df)
        queue.append(processed_tweet_df)


        ##########
      except Exception as e1:
          logging.info("Collection: {}-{} error occurred: {} ".format(start_time_iso, end_time_iso_,e1))
          e1=str(e1)
          email_account.send(to_address,'Data Collection Error',e1) # to_address 'a,b,c',title, content
          break
      if len(queue)%15==0:
        torch.save(queue,'geo_tweets/collected_queue_{}.pt'.format(end_time_))  
        queue =deque()
      
      start_time_ = end_time_
      start_time_iso = end_time_iso_
    torch.save(queue,'geo_tweets/collected_queue_{}.pt'.format(end_time_))    
    logging.info('Collection Done')
    email_account.send(to_address,'Data Collection Alert',"Data collection Finished. \n {}-{} ".format(start_time_iso, end_time_iso)) # to_address 'a,b,c',title, content
    print('Done')




def consumer(queue):
    while True:
      df = queue.get(True)
    
      df_with_pred = mainPredict(df)
      logging.info("{}-{} Prediction Finished".format(start_time_iso, end_time_iso))    
      if queue.empty():

        break
    email_account.send(to_address,'Data Collection Alert',"{}-{} Prediction Finished".format(start_time_iso, end_time_iso)) # to_address 'a,b,c',title, content

    ######  TO DO ########
    # send df_with_pred to database




if __name__ == '__main__':
    # logging setting
    from collections import deque
    queue = deque()
    for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)# Create and configure logger
    logging.basicConfig(filename="datacollection.log",level=logging.INFO,format="%(asctime)s %(message)s")
    
    # send email alert setting
    global email_account
    global to_address
      
    email_account = send_eamil("wpi_uiuc_fs@163.com" ,"HTKSJYOPLAFPQEZL" ) # Don't change this: host_account,password

    to_address = 'rhu@wpi.edu,' #gr-food-safety-swag@wpi.edu,

    # multi_process 
    start_time_iso = '2018-01-01T00:00:00Z' 
    end_time_iso = '2018-02-01T0:00:00Z'


    ###
    producer(queue, start_time_iso, end_time_iso, hasgeo=True) 

