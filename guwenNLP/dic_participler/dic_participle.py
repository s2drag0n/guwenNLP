import pickle


def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                # 优先输出单词长度更长的单词
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        # 提出匹配成功的单词，分词剩余的文本
        i += len(longest_word)
    return word_list


def backward_segment(text, dic):
    word_list = []
    # 扫描位置作为终点
    i = len(text) - 1
    while i >= 0:
        longest_word = text[i]
        for j in range(0, i):
            word = text[j: i + 1]
            if word in dic:
                # 越长优先级越高
                if len(word) > len(longest_word):
                    longest_word = word
                    break
        # 逆向扫描，所以越先查出的单词在位置上越靠后
        word_list.insert(0, longest_word)
        i -= len(longest_word)
    return word_list


def count_single_char(word_list: list):  # 统计单字成词的个数
    return sum(1 for word in word_list if len(word) == 1)


def bidirectional_segment(text, dic):
    f = forward_segment(text, dic)
    b = backward_segment(text, dic)
    print("正向最长匹配:", f)
    print("逆向最长匹配:", b)
    # 词数更少优先级更高
    if len(f) < len(b):
        return f
    elif len(f) > len(b):
        return b
    else:
        # 单字词更少的优先级更高
        if count_single_char(f) < count_single_char(b):
            return f
        else:
            # 词数以及单字词数量都相等的时候，逆向最长匹配优先级更高
            return b


class dic_participle:

    @staticmethod
    def dic_paticiple(text, path_mod):
        with open(path_mod, 'rb') as file:
            dic = pickle.load(file)
        print("双向最长匹配:", bidirectional_segment(text, dic))
