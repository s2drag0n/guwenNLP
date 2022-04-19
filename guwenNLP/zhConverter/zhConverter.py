from guwenNLP.participler.participle import participle
import pickle


def forwardMM(user_dict, sentence):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = max([len(item) for item in user_dict.keys()])
    start = 0
    while start != len(sentence):
        index = start + max_len
        if index > len(sentence):
            index = len(sentence)
        for i in range(max_len):
            if sentence[start:index] in user_dict.keys():
                print(user_dict[sentence[start:index]], end='')
                start = index
                break
            elif len(sentence[start:index]) == 1:
                print(sentence[start:index], end='')
                start = index
                break
            index += -1


class zhConverter:

    @staticmethod
    def zh2hant(s, path_mod_zhc, path_mod_par):
        p = participle()
        with open(path_mod_zhc, 'rb') as file:
            zh2hantDict = pickle.load(file)
        wordList = list(p.participle(s, path_mod_par))
        for item in wordList:
            forwardMM(zh2hantDict, item)
        print("\n")

    @staticmethod
    def zh2hans(s, path_mod_zhc, path_mod_par):
        p = participle()
        with open(path_mod_zhc, 'rb') as file:
            zh2hansDict = pickle.load(file)
        wordList = list(p.participle(s, path_mod_par))
        for item in wordList:
            forwardMM(zh2hansDict, item)
        print("\n")
