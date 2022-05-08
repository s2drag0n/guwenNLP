import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import BertForTokenClassification, BertTokenizer
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm, trange
# from keras.preprocessing.sequence import pad_sequences
import pickle

with open('fifty_data_list.pkl', 'rb') as file1:
    part_data_list = pickle.load(file1)

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

print(part_data_list[0])
all_sentences = []  # 句子
all_labels = []  # labels
for seq_pair in part_data_list:
    if (len(seq_pair[0]) <= 60) & (len(seq_pair[1]) <= 60):
        sentence = "".join(seq_pair[0])
        labels = [tag_to_ix[t] for t in seq_pair[1]]
        all_sentences.append(sentence)
        all_labels.append(labels)
print("all_sentences[0]:{}".format(all_sentences[0]))
print("all_labels[0]:{}".format(all_labels[0]))

BERT_PATH = './guwenbert-base'
# padding
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
tokenized_texts = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_sentences]
model = torch.load('finalModel')
MAX_LEN = 64

for tokenized_text in tokenized_texts:
    for i in range(MAX_LEN - len(tokenized_text)):
        tokenized_text.append(1)
input_ids = tokenized_texts

for label in all_labels:
    label.insert(len(label), 8)  # [SEP]
    label.insert(0, 7)  # [CLS]
    if MAX_LEN > len(label) - 1:
        for i in range(MAX_LEN - len(label)):
            label.append(9)  # [PAD]

for labelss in all_labels:
    if len(labelss) != 64:
        print(len(labelss))

attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i != 1) for i in seq]
    attention_masks.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    all_labels,
                                                                                    random_state=2019,
                                                                                    test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       random_state=2019,
                                                       test_size=0.1)


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


print("Is CUDA available: ", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU numbers: ", n_gpu)
print("device_name: ", torch.cuda.get_device_name(0))


def puc(input_idsss, attention_masksss):
    pred_flat1 = model_use(input_idsss, attention_masksss)[0]
    pred_flat2 = model_use(input_idsss, attention_masksss)[1]
    return pred_flat1, pred_flat2


def compute(pred_flat, i, a):
    global TP, FP, FN
    for j in range(64):
        if list_validation_labels[i][a][j] <= 5:
            if list_validation_labels[i][a][j] == pred_flat[j]:
                TP += 1
            elif pred_flat[j] == 6:
                FN += 1
        elif list_validation_labels[i][a][j] == 6:
            if pred_flat[j] <= 5:
                FP += 1


# tensor化
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
list_validation_inputs = [validation_inputs[i:i + 2] for i in range(0, len(validation_inputs), 2)]
list_validation_labels = [validation_labels[i:i + 2] for i in range(0, len(validation_labels), 2)]
list_validation_masks = [validation_masks[i:i + 2] for i in range(0, len(validation_masks), 2)]
TP = 0
FP = 0
FN = 0
for i in range(len(list_validation_inputs)):
    if len(list_validation_inputs[i]) == 2:
        input_id, attention_masks = gpu_transfer(list_validation_masks[i], list_validation_inputs[i])
        pred_flat1, pred_flat2 = puc(input_id, attention_masks)
        compute(pred_flat1, i, 0)
        compute(pred_flat2, i, 1)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)

print("TP(模型预测与原文中完全匹配的标点数量)", TP)
print("FP(模型预测出来但原标注中没有标点的数量)", FP)
print("FN(原文中有但模型没有预测出来的标点数量)", FN)
print("precision(精确率)", precision)
print("recall(召回率)", recall)
print("F1", F1)
