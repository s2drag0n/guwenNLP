import glob
import os
import pickle
import re

# 删除数据集中的译文
'''
path = "Chinese_dataset/gj/*/*译文.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    os.remove(file)
'''

# 将？！转换为。
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = line.replace("？", "。")
            line = line.replace("！", "。")
            line = line.replace(".", "。")
            file_data += line
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 打印出所有带。“的句子
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            if "。“" in line:
                print(line)
'''

# 将所有的。“转换为。”
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = line.replace("。“", "。”")
            file_data += line
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 将；改为，
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = line.replace("；", "，")
            file_data += line
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 将“，和”，改成。”
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = line.replace("”。", "。”")
            # line = line.replace("”，", "”。")
            file_data += line
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 删除除了  ，   。   ：   。”   ：“   、的所有标点
'''
pattern = re.compile(r'[\u4e00-\u9fa5]+|[，]|[、]|：[“]?|。[”]?')
path = "Chinese_dataset/gj/*/*.txt"
# path = "卷一.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            it = re.finditer(pattern, line)
            for match in it:
                file_data += match.group()
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 在每个。和。”后插入换行符
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = re.sub("。”", "。”\n", line)
            line = re.sub("。(?!”)", "。\n", line)
            file_data += line
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 将标点转换为字母表示
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = line.replace("，", "D")
            line = line.replace("。”", "E")
            line = line.replace("：“", "B")
            line = line.replace("。", "J")
            line = line.replace("、", "P")
            line = line.replace("：", "M")
            file_data += line
    with open(file, "w", encoding='UTF-8') as in_file:
        in_file.write(file_data)
'''

# 生成标记并转化为文件
'''
path = "Chinese_dataset/gj/*/*.txt"
fileList = glob.glob(path, recursive=True)
all_data_list = []
step = 2
pattern = re.compile(r'[\u4e00-\u9fa5]')
for file in fileList:
    file_data = ""
    with open(file, "r", encoding='UTF-8') as in_file:
        for line in in_file:
            line = line.strip()
            zi = []
            mark = []
            list__ = [line[i:i+step] for i in range(0, len(line))]
            for words in list__:
                if len(words) == 2:
                    if re.match(pattern, words[0]):
                        zi.append(words[0])
                        print(zi)
                        if re.match(pattern, words[1]):
                            mark.append("0")
                            print(mark)
                        elif words[1] == "D":
                            mark.append("D")
                            print(mark)
                        elif words[1] == "J":
                            mark.append("j")
                            print(mark)
                        elif words[1] == "M":
                            mark.append("M")
                            print(mark)
                        elif words[1] == "B":
                            mark.append("B")
                            print(mark)
                        elif words[1] == "E":
                            mark.append("E")
                            print(mark)
                        elif words[1] == "P":
                            mark.append("P")
                            print(mark)
            seq_tuple = (zi, mark)
            all_data_list.append(seq_tuple)
# print(all_data_list)

with open("../Punc&Seg/all_data_list.pkl", "wb") as tf1:
    pickle.dump(all_data_list, tf1)
'''

# with open('all_data_list.pkl', 'rb') as file1:
#     all_data_list = pickle.load(file1)
# print(len(all_data_list))

# part_data_list = all_data_list[0:3000]
# with open("../Punc&Seg/part_data_list.pkl", "wb") as tf1:
#     pickle.dump(part_data_list, tf1)

# with open('part_data_list.pkl', 'rb') as file1:
#     part_data_list = pickle.load(file1)
# print(part_data_list)

with open('all_data_list.pkl', 'rb') as file1:
    all_data_list = pickle.load(file1)
print(len(all_data_list))

thirty_data_list = all_data_list[0:300000]
with open("../Punc&Seg/thirty_data_list.pkl", "wb") as tf1:
    pickle.dump(thirty_data_list, tf1)