### 繁简转换

  转为繁体,调用模型`zh2hant.pkl`和分词模型`participle.klm`，模型文件统一在`guwenNLP.modles`中

```Python
from guwenNLP.zhConverter.zhConverter import zhConverter
z = zhConverter()
z.zh2hant("骨董自来多赝，而吴中尤甚，文士皆借以糊口。", "模型路径/zh2hant.pkl", "模型路径/participle.klm")


```

  转为简体,调用模型`zh2hans.pkl`和分词模型`participle.klm`，模型文件统一在`guwenNLP.modles`中

```Python
from guwenNLP.zhConverter.zhConverter import zhConverter
z = zhConverter()
z.zh2hans("骨董自來多贗，而吳中尤甚，文士皆藉以糊口。", "模型路径/zh2hans.pkl", "模型路径/participle.klm")
```