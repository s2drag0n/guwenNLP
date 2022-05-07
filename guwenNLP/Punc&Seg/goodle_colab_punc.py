import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import BertForTokenClassification, BertTokenizer
from transformers import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm, trange
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

model = torch.load('finalModel')

tag_to_ix = {"D": 0,
             "P": 1,
             "j": 2,
             "M": 3,
             "B": 4,
             "E": 5,
             "0": 6,
             "[CLS]": 7,
             "[SEP]": 8,
             "[PAD]": 9}

ix_to_tag = {0: "D",
             1: "P",
             2: "j",
             3: "M",
             4: "B",
             5: "E",
             6: "0",
             7: "[CLS]",
             8: "[SEP]",
             9: "[PAD]"}

tokenizer = BertTokenizer.from_pretrained("ethanyt/guwenbert-base")


def token_maker(text11, text22):
    tokenized_text1 = tokenizer.encode(text11, add_special_tokens=True)
    tokenized_text2 = tokenizer.encode(text22, add_special_tokens=True)
    tokenized_texts = []
    tokenized_texts.append(tokenized_text1)
    tokenized_texts.append(tokenized_text2)
    MAX_LEN = 64
    for tokenized_text in tokenized_texts:
        for i in range(MAX_LEN - len(tokenized_text)):
            tokenized_text.append(1)
    input_ids = tokenized_texts
    input_ids = torch.tensor(input_ids)
    return input_ids, tokenized_texts


def mask_maker(input_ids):
    attention_masks = []
    attention_masks = [[float(i != 1) for i in input_id] for input_id in input_ids]
    attention_masks = torch.tensor(attention_masks)
    return attention_masks


def gpu_transfer(attention_masks, input_ids):
    attention_masks = attention_masks.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    input_ids = input_ids.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return input_ids, attention_masks


def model_use(input_ids, attention_masks):
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_masks,
                    token_type_ids=None,
                    position_ids=None)

    scores = outputs[0].detach().cpu().numpy()
    pred_flat1 = np.argmax(scores[0], axis=1).flatten()
    pred_flat2 = np.argmax(scores[1], axis=1).flatten()
    list = []
    list.append(pred_flat1)
    list.append(pred_flat2)
    return list


def puc_adder(text1, text11, pred_flat1):
    print("原标点：", end="")
    print(text1)
    list_text1 = list(text11)
    flag = 0
    for i in range(len(text1) + 1):
        if (pred_flat1[i] == 0):
            list_text1.insert(i + flag, "，")
            flag += 1
        elif (pred_flat1[i] == 1):
            list_text1.insert(i + flag, "、")
            flag += 1
        elif (pred_flat1[i] == 2):
            list_text1.insert(i + flag, "。")
            flag += 1
        elif (pred_flat1[i] == 3):
            list_text1.insert(i + flag, "：")
            flag += 1
        elif (pred_flat1[i] == 4):
            list_text1.insert(i + flag, "：“")
            flag += 1
        elif (pred_flat1[i] == 5):
            list_text1.insert(i + flag, "。”")
            flag += 1
    print("加标点：", end="")
    print("".join(list_text1))


def puc(a, b):
    text1 = ""
    text2 = ""
    text11 = ""
    text22 = ""
    text1 = a
    text2 = b
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    text11 = re.sub(pattern, '', a)
    text22 = re.sub(pattern, '', b)
    input_idss, tokenized_texts = token_maker(text11, text22)
    attention_maskss = mask_maker(tokenized_texts)
    input_ids, attention_masks = gpu_transfer(attention_maskss, input_idss)
    pred_flat1 = model_use(input_ids, attention_masks)[0]
    pred_flat2 = model_use(input_ids, attention_masks)[1]
    puc_adder(text1, text11, pred_flat1)
    puc_adder(text2, text22, pred_flat2)
