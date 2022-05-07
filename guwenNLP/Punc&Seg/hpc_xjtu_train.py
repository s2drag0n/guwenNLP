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
with open('all_data_list.pkl', 'rb') as file1:
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



MAX_LEN = 64


MAX_LEN = 64
for tokenized_text in tokenized_texts:
  for i in range(MAX_LEN-len(tokenized_text)):
    tokenized_text.append(1)
input_ids = tokenized_texts

# 输入padding
# 此函数在keras里面
# input_ids = pad_sequences([txt for txt in tokenized_texts],
#                           maxlen=MAX_LEN,
#                           dtype="long",
#                           truncating="post",
#                           padding="post",
#                           value=1)

for label in all_labels:
    label.insert(len(label), 8)  # [SEP]
    label.insert(0, 7) # [CLS]
    if MAX_LEN > len(label) -1:
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

# tensor化
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# batch size
batch_size = 16

# 形成训练数据集
train_data = TensorDataset(train_inputs, train_masks, train_labels)
# 随机采样
train_sampler = RandomSampler(train_data)
# 读取数据
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)


# 形成验证数据集
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
# 随机采样
validation_sampler = SequentialSampler(validation_data)
# 读取数据
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForTokenClassification.from_pretrained(BERT_PATH, num_labels=10)
model.cuda()

# BERT fine-tuning parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']

# 权重衰减
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=5e-5)

# 保存loss
train_loss_set = []
# epochs
epochs = 4

print("Is CUDA available: ", torch.cuda.is_available())
n_gpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU numbers: ", n_gpu)
print("device_name: ", torch.cuda.get_device_name(0))

# BERT training loop
for _ in trange(epochs):
    # 训练
    print(f"当前epoch： {_}")
    # 开启训练模式
    model.train()
    tr_loss = 0  # train loss
    nb_tr_examples, nb_tr_steps = 0, 0
    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # 把batch放入GPU
        batch = tuple(t.to(device) for t in batch)
        # 解包batch
        b_input_ids, b_input_mask, b_labels = batch
        # 梯度归零
        optimizer.zero_grad()
        # 前向传播loss计算
        output = model(input_ids=b_input_ids,
                       attention_mask=b_input_mask,
                       labels=b_labels)
        loss = output[0]
        # print(loss)
        # 反向传播
        loss.backward()
        # Update parameters and take a step using the computed gradient
        # 更新模型参数
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print(f"当前 epoch 的 Train loss: {tr_loss/nb_tr_steps}")

# 验证状态
model.eval()

# 建立变量
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
# Evaluate data for one epoch

# 验证集的读取也要batch
# segment embeddings，如果没有就是全0，表示单句
# position embeddings，[0,句子长度-1]
eval_loss = 0
nb_eval_steps = 0
for batch in tqdm(validation_dataloader):
    # 元组打包放进GPU
    batch = tuple(t.to(device) for t in batch)
    # 解开元组
    b_input_ids, b_input_mask, b_labels = batch
    # 预测
    with torch.no_grad():
        outputs = model(input_ids=b_input_ids,
                        attention_mask=b_input_mask,
                        token_type_ids=None,
                        position_ids=None)
        tmp_eval_loss = model(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              labels=b_labels)
        nb_eval_steps += 1
        eval_loss += tmp_eval_loss[0]
        # print(logits[0])
    # Move logits and labels to CPU
    scores = outputs[0].detach().cpu().numpy()  # 每个字的标签的概率
    pred_flat = np.argmax(scores[0], axis=1).flatten()
    label_ids = b_labels.to('cpu').numpy()  # 真实labels
    # print(logits, label_ids)
eval_loss = eval_loss/nb_eval_steps
print("Validation loss: {}".format(eval_loss))
torch.save(model, "finalModel")