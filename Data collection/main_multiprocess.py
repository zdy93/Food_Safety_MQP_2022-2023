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

    start_time_ = datetime.datetime.strptime(start_time_iso, '%Y-%m-%dT%H:%M:%SZ')
    end_time = datetime.datetime.strptime(end_time_iso, '%Y-%m-%dT%H:%M:%SZ')
    time_interval = 60*12   # 24 Hours, in minutes

    while start_time_< end_time:

        end_time_ = start_time_ + datetime.timedelta(hours=0,minutes = time_interval)
        end_time_iso_ = datetime.datetime.strftime(end_time_,'%Y-%m-%dT%H:%M:%SZ')
        if end_time_ > end_time:
            end_time_ = end_time
            end_time_iso_ = end_time_iso
        try:    
          ##########
          processed_tweet_df = mainCollect(start_time_iso, end_time_iso_, hasgeo=hasgeo)
          # logging.info('\r {}-{} Got Processed Tweets:  {}'.format(start_time_iso, end_time_iso_,len(processed_tweet_df)))
          ## send into queue (pass to consumer)
          queue.put((processed_tweet_df))

          start_time_ = end_time_
          start_time_iso = end_time_iso_
          ##########
        except Exception as e1:
            logging.info("Collection: {}-{} error occurred: {} ".format(start_time_iso, end_time_iso_,e1))
            e1=str(e1)
            email_account.send(to_address,'Data Collection Error',e1) # to_address 'a,b,c',title, content
            break    




def consumer(queue):
    df = queue.get(True)
    df_with_pred = mainPredict(df)
    

    ######  TO DO ########
    # send df_with_pred to database




if __name__ == '__main__':
    # logging setting
    for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)# Create and configure logger
    logging.basicConfig(filename="datacollection.log",level=logging.INFO,format="%(asctime)s %(message)s")
    
    # send email alert setting
    global email_account
    global to_address
      
    email_account = send_eamil("wpi_uiuc_fs@163.com" ,"HTKSJYOPLAFPQEZL" ) # Don't change this: host_account,password
    to_address = 'gr-food-safety-swag@wpi.edu, rhu@wpi.edu'

    # multi_process 
    start_time_iso = '2018-01-01T00:00:00Z' 
    end_time_iso = '2018-01-02T0:00:00Z'

    logging.info('main process is {}'.format(os.getpid()))
    logging.info('core number is {}'.format(cpu_count()))
    st = time.time()
    queue = Queue()

    try:
      p1 = Process(target=producer, args=(queue, start_time_iso, end_time_iso,True)) ## hasgeo=True
      email_account.send(to_address,'Data Collection Alert',"{}-{} Data collection Finished".format(start_time_iso, end_time_iso)) # to_address 'a,b,c',title, content
    
    except:
      pass
    try:
      p2 = Process(target=consumer,args=(queue,))
      email_account.send(to_address,'Data Collection Alert',"{}-{} Prediction Finished".format(start_time_iso, end_time_iso)) # to_address 'a,b,c',title, content
      logging.info("{}-{} Prediction Finished".format(start_time_iso, end_time_iso))
    except Exception as e2:
      logging.info("Prediction occurred:{}".format(e2))
      e2=str(e2)
      email_account.send(to_address,'Prediction Error',e2) # to_address 'a,b,c',title, content



    p1.start()
    p2.start()
    p1.join() ## if p1 hasn't been completed, the main process won't continue (the start_time/end time won;t be changed) 
    et = time.time()
    
    logging.info('total time is {} s'.format(et-st))
