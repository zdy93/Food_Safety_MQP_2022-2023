# Model Prediction Demo

Codes for using trained Roberta model to make prediction on collected tweet data.

## Setup

Python version: 3.8.13

Run the code below to install modules.
```linux

```
If TorchCRF cannot be installed by the command above. Use the command below to install modules.
```linux
cat requirements.txt | xargs -n 1 pip install
pip install pytorch-crf
```
Download the model from the usda-foodpoisoning server. The directory is ```/home/cgnoreika/pytorch_model.bin```

Download the ```tmp``` directory and create an ```output``` directory in your machine. Or you can use the pipeline code to get some tweets for prediction. 

## Running
1. Modify the input_dir and model_dir variables in the ```main_prediction.py``` to match your local directories
2. Run the code below
```linux
python main_prediction.py
```
3. Now you can find output file in the ```output``` folder.
