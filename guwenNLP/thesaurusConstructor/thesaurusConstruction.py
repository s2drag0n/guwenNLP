import time
from guwenNLP.thesaurusConstructor import dataProcess
from math import log2
import ntpath


# 字典树类
class Trie:
    # 结点类
    class TrieNode:
        def __init__(self):
            self.freq = 0  # 词频
            self.pmi = 0  # 点互信息
            self.r_entropy = 0  # 右熵
            self.l_entropy = 0  # 左熵
            self.children = {}  # 孩子结点

    # 初始化
    def __init__(self):
        self.root = self.TrieNode()

    # 添加结点
    def add(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.freq += 1

    # 查询结点
    def find(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node


# 词库构建类
class thesaurusConstruction:
    # 定义最小词长和最大字长
    MIN_WORD_LEN = 1
    MAX_WORD_LEN = 4

    # TODO: Different PMI and Entropy thresholds（阈值） for different lengths
    MIN_WORD_FREQ = 10
    MIN_PMI = 80
    MIN_ENTROPY = 2

    # 初始化词库构建类
    def __init__(self):
        self.trie = Trie()  # 正向字典树
        self.r_trie = Trie()  # 逆向字典树
        self.total = 0  # 词总数

    # 词库构建函数
    def construct_thesaurus(self, data_file):
        self.build_trie_trees(data_file)  # 根据文件构建双字典树
        self.compute()  # 计算点互信息和左右临界熵
        corpus = self.filter()  # 根据上一步计算结果进行过滤？
        return corpus

    def build_trie_trees(self, data_file):
        """ Counts frequency of segments of data, also records their left and right char sets.
            计算词语的频数，并且记录他们的左右字符集。
        """
        max_seg_len = self.MAX_WORD_LEN + 1

        start = time.time()
        for text in dataProcess.text_iterator(data_file):
            length = len(text)
            for i in range(length):
                for j in range(1, min(length - i + 1, max_seg_len + 1)):
                    seg = text[i: i + j]
                    self.trie.add(seg)

                    r_seg = seg[::-1]
                    self.r_trie.add(r_seg)

                    self.total += 1
        end = time.time()

        print('Trie building time:', end - start)

    def compute(self):
        start = time.time()
        node = self.trie.root
        word = ''
        self.compute_help(node, word)
        end = time.time()
        print('Computation time:', end - start)

    def compute_help(self, node, word):
        if node.children:
            for char, child in node.children.items():
                word += char
                if len(word) <= self.MAX_WORD_LEN:
                    self.calculate_pmi(child, word)
                    self.calculate_rl_entropy(child, word)
                    self.compute_help(child, word)
                word = word[:-1]  # 删除最后一个字符

    def calculate_pmi(self, node, word):
        length = len(word)
        if length == 1:
            node.pmi = self.MIN_PMI
        else:
            constant = node.freq * self.total
            mutuals = (constant / (self.trie.find(word[:i + 1]).freq * self.trie.find(word[i + 1:]).freq)
                       for i in range(length - 1))
            node.pmi = min(mutuals)

    def calculate_rl_entropy(self, node, word):
        # right entropy
        if node.children:
            node.r_entropy = self.calculate_entropy(node)
        # left entropy
        r_word = word[::-1]
        r_node = self.r_trie.find(r_word)
        if r_node.children:
            node.l_entropy = self.calculate_entropy(r_node)

    @staticmethod
    def calculate_entropy(node):
        freqs = [child.freq for child in node.children.values()]
        sum_freqs = sum(freqs)
        entropy = sum([- (x / sum_freqs) * log2(x / sum_freqs) for x in freqs])
        return entropy

    # 过滤掉不合适的词
    def filter(self):
        """ Filters the PMI and entropy calculation result dict, removes words that do not
            reach the thresholds.
            TODO: test use max of r/l entropy to filter.
        """
        start = time.time()
        node = self.trie.root
        word = ''
        word_dict = {}
        self.filter_help(node, word, word_dict)
        end = time.time()
        print('Word filtering:', end - start)
        return word_dict

    def filter_help(self, node, word, word_dict):
        if node.children:
            for char, child in node.children.items():
                word += char
                if self.valid_word(child, word):
                    word_dict[word] = [child.freq, child.pmi, child.r_entropy, child.l_entropy]
                self.filter_help(child, word, word_dict)
                word = word[:-1]

    # 判断词频、pmi、左右临界熵是否满足要求
    def valid_word(self, node, word):
        if self.MIN_WORD_LEN <= len(word) <= self.MAX_WORD_LEN \
                and node.freq >= self.MIN_WORD_FREQ \
                and node.pmi >= self.MIN_PMI \
                and node.r_entropy >= self.MIN_ENTROPY \
                and node.l_entropy >= self.MIN_ENTROPY:
            # and not self.has_stopword(word):
            return True
        return False

    '''
    def has_stopword(self, word):
        """ Checks if a word contains stopwords, which are not able to construct words.
        """
        if len(word) == 1:
            return False
        for char in word:
            if char in stopchars:
                return True
        return False
    '''

    @staticmethod
    def save(corpus, out_f):
        """ Saves the word detection result in a csv file.
        """
        words = sorted(corpus, key=lambda x: (len(x), -corpus[x][0], -corpus[x][1], -corpus[x][2], -corpus[x][3]))
        with open(out_f, 'w') as f:
            f.write('Word,Frequency,PMI,R_Entropy,L_Entropy\n')
            for word in words:
                f.write('{},{},{},{},{}\n'.format(
                        word, corpus[word][0], corpus[word][1],
                        corpus[word][2], corpus[word][3]))

    @staticmethod
    def corpusConstruction(file):
        file_name = ntpath.basename(file)
        theConstructor = thesaurusConstruction()
        corpus = theConstructor.construct_thesaurus(file)
        theConstructor.save(corpus, file_name[:-4] + "_词库.txt")
