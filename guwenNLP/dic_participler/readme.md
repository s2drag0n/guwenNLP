#### 字典分词（词典使用“儒藏”（400MB文本文件）进行构建）

调用模型`儒藏.pkl`，模型文件统一在`guwenNLP.modles`中

```Python
from guwenNLP.dic_participler.dic_participle import dic_participle 
d = dic_participle()
d.dic_paticiple("骨董自来多赝，而吴中尤甚，文士皆借以糊口。", "../modles/儒藏.pkl")

```