# guwenNLP

### 词库构建

```Python
from guwenNLP.thesaurusConstructor.thesaurusConstruction import thesaurusConstruction;\
t = thesaurusConstruction();\
path = 'guwenNLP/thesaurusConstructor/乐书.txt';\
t.corpusConstruction(path)
```

### 分词

#### 基于隐马尔可夫模型、n-gram算法、viterbi算法的无监督分词

```
调用模型`participle.klm`
```

```Python
from guwenNLP.participler.participle import participle;\
p = participle();\
result = p.participle("骨董自来多赝，而吴中尤甚，文士皆借以糊口。", "guwenNLP/modles/participle.klm");\
print(result)
from guwenNLP.participler.participle import participle;\
p = participle();\
result = p.participle("先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。", "guwenNLP/modles/participle.klm");\
print(result)
```

```
![](https://secure2.wostatic.cn/static/pR5ZUM4SzDdRZFwN8RkfEr/image.png)
```

#### 字典分词（词典使用“儒藏”（400MB文本文件）进行构建）

```
调用模型`儒藏.pkl`
```

```Python
from guwenNLP.dic_participler.dic_participle import dic_participle 
d = dic_participle()
d.dic_paticiple("骨董自来多赝，而吴中尤甚，文士皆借以糊口。", "guwenNLP/modles/儒藏.pkl")
from guwenNLP.dic_participler.dic_participle import dic_participle 
d = dic_participle()
d.dic_paticiple("先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。", "guwenNLP/modles/儒藏.pkl")
```

```
![](https://secure2.wostatic.cn/static/81SiesHUSbwJKJa8FLZ2Fg/image.png)
```

### 繁简转换

转为繁体,调用模型`zh2hant.pkl`和分词模型`participle.klm`

```Python
from guwenNLP.zhConverter.zhConverter import zhConverter
z = zhConverter()
z.zh2hant("骨董自来多赝，而吴中尤甚，文士皆借以糊口。", "guwenNLP/modles/zh2hant.pkl", "guwenNLP/modles/participle.klm")
from guwenNLP.zhConverter.zhConverter import zhConverter
z = zhConverter()
z.zh2hant("先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。", "guwenNLP/modles/zh2hant.pkl", "guwenNLP/modles/participle.klm")
```

转为简体,调用模型`zh2hans.pkl`和分词模型`participle.klm`

```Python
from guwenNLP.zhConverter.zhConverter import zhConverter
z = zhConverter()
z.zh2hans("骨董自來多贗，而吳中尤甚，文士皆藉以糊口。", "guwenNLP/modles/zh2hans.pkl", "guwenNLP/modles/participle.klm")
from guwenNLP.zhConverter.zhConverter import zhConverter
z = zhConverter()
z.zh2hans("先帝創業未半而中道崩殂，今天下三分，益州疲弊，此誠危急存亡之秋也。", "guwenNLP/modles/zh2hans.pkl", "guwenNLP/modles/participle.klm")
```

### 自动加标点&断句

基于[GuwenBERT]([GitHub - Ethan-yt/guwenbert: GuwenBERT: 古文预训练语言模型（古文BERT） A Pre-trained Language Model for Classical Chinese (Literary Chinese)](https://github.com/Ethan-yt/guwenbert))预训练模型训练的`finalModel`进行自动加标点并断句，调用需配置torch和tensorflow并在GPU上进行使用。

其中`Punc&Seg`目录下的`hpc_xjtu_train.py`是finalModel的训练脚本；`all_data_list.pkl`是训练使用语料，使用[杨钊师兄的文言文语料库]([GitHub - zhaoyang9425/modern-ancient_Chinese_dataset: a primary modern-ancient Chinese dataset](https://github.com/zhaoyang9425/modern-ancient_Chinese_dataset)[GitHub - zhaoyang9425/modern-ancient_Chinese_dataset: a primary modern-ancient Chinese dataset](https://github.com/zhaoyang9425/modern-ancient_Chinese_dataset))清洗处理得到，包含3809001条数据，其中数据处理代码为`data_process.py`；

`modle调用.ipynb`是在Google Colab平台调用`finalModel`给输入文本自动加标点的脚本，只需调用`punc(text1,text2)`函数即可（因为BERT模型要求至少两个输入，如果只有一段需要加标点的文本可直接将text2置为空字符串：`punc(text1,"")`）。

### 模型链接

模型文件太大，无法上传到github，百度云盘链接如下

<mark>链接：https://pan.baidu.com/s/1kRgQe5UppoTvsUJYV0CWSw 
提取码：yyyy </mark>