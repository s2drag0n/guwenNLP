#### 基于隐马尔可夫模型、n-gram算法、viterbi算法的无监督分词  

调用模型`participle.klm`，模型文件统一在`guwenNLP.modles`中

```Python
from guwenNLP.paticipler.participle import participle
p = participle()
result = p.participle("骨董自来多赝，而吴中尤甚，文士皆借以糊口。", "模型路径/participle.klm")
print(result)
```