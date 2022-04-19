# -*- coding = =utf-8 -*-
# @Time : 2022/4/10 22:03
# @Author : 宋子龙
# @File : __init__.py.py
# @Software : PyCharm

import kenlm


def load_lm_participle(path_mod):
    return kenlm.LanguageModel(path_mod)
