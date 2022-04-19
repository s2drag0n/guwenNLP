from math import log10
from guwenNLP import modles

# 这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
trans = {'bb': 1, 'bc': 0.15, 'cb': 1, 'cd': 0.01, 'db': 1, 'de': 0.01, 'eb': 1, 'ee': 0.001}
trans = {i: log10(j) for i, j in trans.items()}


class participle:

    @staticmethod
    def vtb(nodes):
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_ = paths
            paths = {}
            for i in nodes[l]:
                nows = {}
                for j in paths_:
                    if j[-1] + i in trans:
                        nows[j + i] = paths_[j] + nodes[l][i] + trans[j[-1] + i]
                k = list(nows.values()).index(max(nows.values()))
                paths[list(nows.keys())[k]] = list(nows.values())[k]
        return list(paths.keys())[list(paths.values()).index(max(paths.values()))]

    def cp(self, s, model):
        return (model.score(' '.join(s), bos=False, eos=False) - model.score(' '.join(s[:-1]), bos=False,
                                                                             eos=False)) or -100.0

    def participle(self, s, path_mod):
        model = modles.load_lm_participle(path_mod)
        nodes = [{'b': self.cp(s[i], model), 'c': self.cp(s[i - 1:i + 1], model), 'd': self.cp(s[i - 2:i + 1], model),
                  'e': self.cp(s[i - 3:i + 1], model)} for i in
                 range(len(s))]
        tags = self.vtb(nodes)
        words = [s[0]]
        for i in range(1, len(s)):
            if tags[i] == 'b':
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
