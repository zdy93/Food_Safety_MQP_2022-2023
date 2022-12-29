from transformers import AdamW, AutoTokenizer, RobertaConfig
import pandas as pd
import numpy as np
# from dotmap import DotMap
import time
import json
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
from ast import literal_eval
from tqdm import tqdm
import json
import argparse
import logging
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import glob
import os
import model_weighted_roberta
import utils

import logging
log = logging.getLogger(__name__)

def simple_tokenize_no_label(orig_tokens, tokenizer, max_seq_length):
    """
    tokenize a array of raw text
    """
    # orig_tokens = orig_tokens.split()

    fake_label_id = 0
    pad_token_label_id = -100
    tokens = []
    label_ids = []
    for word in orig_tokens:
        word_tokens = tokenizer.tokenize(word)

        # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
        if len(word_tokens) > 0:
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([fake_label_id] + [pad_token_label_id] * (len(word_tokens) - 1))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[: (max_seq_length - special_tokens_count)]
        label_ids = label_ids[: (max_seq_length - special_tokens_count)]
    bert_tokens = [tokenizer.cls_token]
    # bert_tokens = ["[CLS]"]
    bert_tokens.extend(tokens)
    label_ids = [pad_token_label_id] + label_ids
    bert_tokens.append(tokenizer.sep_token)
    # bert_tokens.append("[SEP]")
    label_ids += [pad_token_label_id]
    return bert_tokens, label_ids


def tokenize_with_new_mask_no_label(orig_text, max_length, tokenizer):
    """
    tokenize a array of raw text and generate corresponding
    attention labels array and attention masks array
    """
    pad_token_label_id = -100
    simple_tokenize_results = [list(tt) for tt in zip(
        *[simple_tokenize_no_label(orig_text[i], tokenizer, max_length) for i in
          range(len(orig_text))])]
    bert_tokens, label_ids = simple_tokenize_results[0], simple_tokenize_results[1]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in bert_tokens]
    input_ids = utils.pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    label_ids = utils.pad_sequences(label_ids, maxlen=max_length, dtype="long", truncating="post", padding="post",
                              value=pad_token_label_id)
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)
    return input_ids, attention_masks, label_ids
    
    
def predict(model, test_batch_generator, num_batches, device, token_lambda_arg, label_map):
    model.eval()
    output_t_pred, output_s_pred = None, None
    output_tag = []
    with torch.no_grad():
        for b in tqdm(range(num_batches)):
            x_batch, t_batch, masks_batch = next(test_batch_generator)
            x_batch = Variable(torch.LongTensor(x_batch)).to(device)
            masks_batch = Variable(torch.FloatTensor(masks_batch)).to(device)
            t_batch = t_batch.astype(np.float)
            t_batch = Variable(torch.LongTensor(t_batch)).to(device)
            
            token_weight = None
            y_weight = None

            outputs = model(input_ids=x_batch, attention_mask=masks_batch, token_labels=t_batch,
                            token_class_weight=token_weight, seq_class_weight=y_weight, token_lambda=token_lambda_arg)

            loss, token_logits, seq_logits = outputs[:3]
            if output_t_pred is None:
                output_t_pred = token_logits.detach().cpu().numpy()
                output_s_pred = seq_logits.detach().cpu().numpy()
            else:
                output_t_pred = np.concatenate([output_t_pred,token_logits.detach().cpu().numpy()],axis=0)
                output_s_pred = np.concatenate([output_s_pred,seq_logits.detach().cpu().numpy()],axis=0)
            if type(model) is model_weighted_roberta.RobertaForTokenAndSequenceClassificationWithCRF:
                output_tag.extend(outputs[3])
        return output_t_pred, output_s_pred, output_tag
        
        
def multi_batch_seq_predict_generator(X, token_label, masks, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]

    i = 0
    while True:
        if masks is not None:
            yield X[i:i + batch_size], token_label[i:i + batch_size], masks[i:i + batch_size]
        else:
            yield X[i:i + batch_size], token_label[i:i + batch_size]
        if i + batch_size >= size:
            break
        else:
            i += batch_size
            

def align_predictions(predictions, label_ids, label_map):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    preds = preds.tolist()
    label_map_switch = {label_map[k]: k for k in label_map}
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i][j] != torch.nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map_switch[label_ids[i][j].item()])
                preds_list[i].append(label_map_switch[preds[i][j]])

    return preds_list, out_label_list


def get_sentence_prediction(preds):
    '''
    Returns predicted labels and probabilities given logits
    '''
    m = nn.Softmax(dim=1)
    probabilities = m(torch.tensor(preds))
    y_values, indices = torch.max(probabilities, 1)
    y_label_pred = indices
    return y_label_pred.numpy(), y_values.numpy()
    
    
def load_model(model_type, model_path, config):
    if model_type.startswith('bertweet-multi') and not model_type.startswith('bertweet-multi-crf'):
        model = model_weighted_roberta.RobertaForTokenAndSequenceClassification.from_pretrained(model_path, config=config)

    elif model_type == 'bertweet-multi-crf':
        model = model_weighted_roberta.RobertaForTokenAndSequenceClassificationWithCRF.from_pretrained(model_path, config=config)

    else:
        model = None

    return model
    
    
### Main
def mainPredict(us_data):
    "the input 'us_data' is a dataframe"
    # input_dir = "tmp/"

    # path = os.path.join(input_dir, 'tweets_*.txt_chunk-*.csv')
    # file_list = glob.glob(path)
    # print(file_list)

    # fcount = 0

    # for file_name in file_list:
    bert_model = "roberta-base"
    model_type = "bertweet-multi-crf"
    model_dir = "./pytorch_model.bin"

    task_type = 'entity_detection'
    n_epochs = 10
    max_length = 128
    rnn_hidden_size = 384
    batch_size = 500
    eval_batch_size = 500
    test_batch_size = 500
    seed = 42
    learning_rate = 1e-5
    data = 'wnut_16'
    log_dir = './'
    save_model = False
    early_stop = False
    assign_token_weight = False
    assign_seq_weight = False
    token_lambda = 10
    performance_file = "all_test_performance.txt"

    assert model_type.startswith('bertweet')
    assert task_type in ['entity_detection', 'relevant_entity_detection', 'entity_relevance_classification']

    log_directory = "./" 
    log_filename = 'log.txt'
    
    logname = log_directory + log_filename
    
    

    us_data.dropna(subset=['tweet_token'],inplace=True)
    us_data.reset_index(drop=True,inplace=True)
    # print(f"Total Sentences: {us_data.shape[0]}")
    # X_test_raw = us_data['tweet_token'].apply(literal_eval)
    X_test_raw = us_data['tweet_token']
    tokenizer = AutoTokenizer.from_pretrained(bert_model, normalization=True)
    label_map = {"O": 0, "B-food": 1, "I-food": 2, "B-symptom": 3, "I-symptom": 4, "B-loc": 5, "I-loc": 6, "B-other": 7, "I-other": 8}
    label_map_switch = {label_map[k]: k for k in label_map}
    labels = list(label_map.keys())

    device = torch.device("cpu")
    
    config = RobertaConfig.from_pretrained(bert_model)
    config.update({'num_token_labels': len(labels), 'num_labels': 2,
                'token_label_map': label_map,})
    
    model = load_model(model_type, model_dir, config)
    model = model.to(device)

    X_test, masks_test, token_label_test  = tokenize_with_new_mask_no_label(X_test_raw, max_length, tokenizer)
    num_batches = np.int(np.ceil(X_test.shape[0] / test_batch_size))
    test_batch_generator = multi_batch_seq_predict_generator(X_test, token_label_test, masks_test, test_batch_size)

    a = time.time()
    test_t_pred, test_s_pred, preds_list = predict(model, test_batch_generator, num_batches, device,
                                                token_lambda, label_map)
    b = time.time()
    print(f"Prediction time is {b-a}s")
    
    if type(model) is not model_weighted_roberta.RobertaForTokenAndSequenceClassificationWithCRF:
        preds_list, _ = align_predictions(test_t_pred, token_label_test, label_map)

    test_s_l_pred, test_s_p_pred = get_sentence_prediction(test_s_pred)
    preds_list = pd.Series(preds_list)
    preds_list = preds_list.apply(lambda x: ','.join([label_map_switch[i] for i in x]))
    us_data['token_prediction'] = preds_list
    us_data['sentence_prediction']= test_s_l_pred
    us_data['sentence_prediction_prob'] = test_s_p_pred
    pos_count = us_data['sentence_prediction'].sum()

    # us_data.to_csv("output/predicted_tweets_" + str(fcount) + ".csv", index=False)
    # fcount += 1

    logging.info(f"Total Sentences: {us_data.shape[0]}, Predicted Related Sentences: {pos_count}")
    return us_data  

    
if __name__ == '__main__':
    us_data =pd.read_csv('')
    predicion = mainPredict(us_data)