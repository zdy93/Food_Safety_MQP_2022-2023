# Description
This directory includes all scripts for data collection and prediction. You need to put the model "pytorch_model.bin" into this directory.
The path on the server is ```/home/cgnoreika/pytorch_model.bin```.


## Data collection
```main_collection.py``` is for data collection ***only*** . It will save the collected tweets into ```*.pt``` files in the geo_tweets folder.  
Run code below to start collecting. You need to change the start_time_iso and end_time_iso.
```linux
python main_collection.py
``` 
It will save the log info into `logs/datacollection.log`


## Tweet Prediction
```main_predict.py``` is for prediction ***only***. It will read all the ```*.pt``` files in the geo_tweets folder and do the prediction. 

:exclamation:**To do: Connect to your database and save the prediction into tables**
```linux
python main_predict.py
``` 
It will save the log info into `logs/Predcition.log`

## Whole pipeline (Don't touch this until the database part is ready)

Run the code below for data collection and prediction parallelly.
```linux
python main_multiprocess.py
``` 
It will save the log info into `logs/whole_pipeline.log`

## Send email
It will send us emails when the data collection/prediction is done or it occurs some errors. You can change the receivers by editing the variable `to_address` in the main_.py files.


# Tasks
**I will start the data collection for tweets with geo on the server. All the collected tweets will be saved into folder `home/rhu/geo_tweets`**
1. Complete the database part in 'main_prediction.py'
2. Run 'python main_prediction.py' on the server. (For now, it will take the files in geo_tweets_test as input)
