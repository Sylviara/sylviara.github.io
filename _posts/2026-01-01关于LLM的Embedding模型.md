---
layout:     post
title:      sentence-transformer中的LLM-Embedding模型
subtitle:   到底会不会tokenize？答案是：会的。
date:       2026-01-01
co-author:     ZHAIYU
header-img: img/tqdm_20240410133238.png
catalog: true
tags:
    - pytorch
    - sentence-transformer
---

# 关于LLM的Embedding模型

这个问题起始于我在词汇聚类的时候发现我之前使用的embedding方法都是直接sentence-transformer通过model.encode_query提供embeddings，而Monica给出的例子却使用BERT embedding的tokenizer先批次tokenize我的词汇数据，然后再用model去encode tokenize之后的数据。在解释这样的行为的时候涉及到了pooling的方法，因为我是按照词汇列表的方式表示隶属于同一标准词的输入数据，因此如果直接tokenize再以sentence形式送入model的时候，模型做embedding就是把同一组列表里的词汇按次序组合起来，输出embedding然后取[CLS] token作为所有词汇的归一化表征，但是也有一种方法是对所有词汇都进行embedding然后对每一个单独的词汇进行归一化表征。这时候就产生了一个问题，sentence_transformer提供的model.encode入口和我们对bert词嵌入模型先tokenize再model表征的方法有什么区别？



因此我问了Monica这样一些问题：

> tsne降维和pca降维的区别是什么？优劣分别是？什么时候应该用pca什么时候用tsne？我这种用embedding表示的文本数据使用的降维方法最好选择哪个？以及Qwen很明显是一个利用encoder-only架构的embedding，它和其他基于encoder-decoder的transformer架构的embedding有什么区别？是否也有CLS这个token?是否需要我对输入的词汇进行tokenize之后再取CLStoken的表征？tokenize再model之后取CLS的embedding方法和sentence_transformer里面的encode_query有什么本质区别吗，这两个embedding方法一样吗？

Monica这样回答：

>### t-SNE 降维与 PCA 降维的区别
>
>**1. 原理**：
>
>- PCA（主成分分析）
>  - PCA 是一种线性降维技术，通过寻找数据中的主成分（即方差最大的方向）来减少维度。它通过特征协方差矩阵的特征值分解来实现，保留数据中最重要的线性特征。
>- t-SNE（t-分布随机邻域嵌入）
>  - t-SNE 是一种非线性降维技术，主要用于可视化高维数据。它通过将高维空间中相似的数据点在低维空间中保持相对距离，强调局部结构，适合展示复杂的数据分布。
>
>**2. 优劣**：
>
>- **PCA**：
>  - 优点
>    - 计算速度快，适合大规模数据。
>    - 结果易于解释（主成分的线性组合）。
>  - 缺点
>    - 只能捕捉线性关系，无法处理复杂的非线性结构。
>    - 对于噪声敏感。
>- **t-SNE**：
>  - 优点
>    - 能够很好地捕捉数据的非线性结构，适合可视化高维数据。
>    - 常用于数据探索和可视化，能清晰展示聚类和分布。
>  - 缺点
>    - 计算复杂度高，处理大数据集时可能非常慢。
>    - 结果的解释性较差，难以从低维空间推断高维空间的结构。
>
>### 何时使用 PCA 和 t-SNE
>
>- **使用 PCA**：
>  - 当你需要快速降维并保持数据的全局结构时，尤其是数据是线性可分的情况。
>  - 对于大规模数据集，PCA 是一个更好的选择。
>- **使用 t-SNE**：
>  - 当你需要可视化高维数据，尤其是当数据可能包含复杂的非线性结构时。
>  - 适合于小到中等规模的数据集，通常用于数据探索阶段。
>
>### 对于文本数据的降维选择
>
>对于使用嵌入表示的文本数据，选择降维方法通常取决于你的目标：
>
>- **如果目标是可视化**，那么使用 t-SNE 更合适，因为它能够更好地展示文本数据中的局部结构和聚类。
>- **如果目标是进行后续分析**（如特征选择、分类等），那么使用 PCA 可能更合适，因为它保留了数据的全局结构并且计算效率高。
>
>### Qwen 的嵌入与其他 Transformer 架构的区别
>
>**1. Encoder-only 架构与 Encoder-Decoder 架构**：
>
>- **Encoder-only 架构**（如 Qwen）：
>  - 主要用于生成文本的表示，如句子嵌入。它通常只关注输入的上下文，并生成一个固定大小的嵌入向量。
>  - 在这种架构中，嵌入通常是通过对输入序列的所有 token 的表示进行池化（如取平均或取 CLS token）来获得。
>- **Encoder-Decoder 架构**（如 BERT、T5）：
>  - 这种架构包含编码器和解码器，适合用于生成任务（如翻译、文本生成）。
>  - 通常在解码器中会用到先前生成的 token 信息。【就很迷，先前和输入序列的区别是什么？】
>
>**2. CLS Token**：
>
>- 在许多 Transformer 模型中（如 BERT），CLS token 用于表示整个序列的嵌入。这个 token 的表示通常用于分类任务。
>
>- 在 Qwen 这样的 encoder-only 架构中，是否使用 CLS token 取决于具体的实现和任务要求。
>
>  【还是很迷，所以Encoder-only架构是否还采用CLS token呢？CLS不是class的缩写吗？如果不是分类任务是不是就不用CLS作为序列的开头，不用[SEP]作为序列的结尾呢？】
>
>### Tokenize 和 CLS Token 的使用
>
>- **Tokenization**：
>  - 在输入数据之前，需要将文本进行 tokenization，以将其转换为模型可以处理的格式。对于 Qwen 和其他 Transformer 模型，这一步是必要的。
>- **取 CLS Token 的表征**：
>  - 如果模型设计中包含 CLS token，通常可以直接使用其表示作为整个序列的嵌入。
>  - 取 CLS token 的方法与使用 `encode_query` 的方法可能有本质区别，具体取决于模型的实现和训练目标。
>
>### Tokenize 后取 CLS 的表征与 `encode_query` 的区别
>
>- **Tokenize 后取 CLS 的表征**：
>  - 这是将文本输入到模型后，获取 CLS token 的表示，通常用于分类任务。
>- **`encode_query` 方法**：
>  - 这是特定于某些模型的接口，用于获取输入文本的嵌入，通常会考虑上下文信息，并可能使用不同的池化策略。
>
>### 总结
>
>- **降维选择**：对于文本数据，t-SNE 更适合可视化，PCA 更适合快速分析。
>- **嵌入架构**：Qwen 和其他 Transformer 架构在设计和用途上有明显区别，选择使用 CLS token 或其他嵌入方式取决于具体任务。
>- **Tokenization 和嵌入**：在使用模型之前，tokenization 是必需的，获取嵌入的方法取决于模型的具体实现。

我们单独拿这一个很迷惑的回答：

> 在 Qwen 这样的 encoder-only 架构中，是否使用 CLS token 取决于具体的实现和任务要求。

【还是很迷，所以Encoder-only架构是否还采用CLS token呢？CLS不是class的缩写吗？如果不是分类任务是不是就不用CLS作为序列的开头，不用[SEP]作为序列的结尾呢？】

Monica终究没能解决我的疑问，因此我去encode源码找到了内部的具体embedding信息，得出结论就是，输入只能是list of string，如果输入变成了list of list of string，那么每一个小的list将会被视作一句话，包括其中的逗号。
首先是[源代码](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/SentenceTransformer.py#L446-L573)第614行已经约束，输入务必要么是string，要么是list of string，不可以有其他情况。

```sentences (Union[str, List[str]]): The sentences to embed.```

此外也有[issue](https://github.com/huggingface/sentence-transformers/issues/3278)提到了单个输入和分批次输入的时候导致的embedding的微妙差别在e08级别，就是差距非常非常非常小，但是不可否认它是存在的，毕竟每一次embedding的结果都是黑盒随机来的。



为了解决我的疑问，我亲自把sentence_transformer的源码1057-1062行拿来改造一下并且实验：

```python
#源代码：
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch, **kwargs)

```
改造后的代码输出结果显示对列表里文字的编码完全是正确的：

```python
#改进后的实验代码
from tqdm.autonotebook import trange
def text_length(text) -> int:
    if isinstance(text, dict):  # {key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, "__len__"):  # Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
        return len(text)
    else:
        return sum([len(t) for t in text])  # Sum of length of individual strings
sentences = ["我吃屎","我吃粪","我吃大便"]
# sentences = [sentences,sentences,sentences]
length_sorted_idx = np.argsort([-text_length(sen) for sen in sentences])
sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

for start_index in trange(0, len(sentences), 1, desc="Batches", disable=not True):
    sentences_batch = sentences_sorted[start_index : start_index + 2]
    features = model.tokenize(sentences_batch)
    # 从input_ids恢复原始sentence词汇：
    sentences_recovered = []
    for input_id in features['input_ids']:
        tokens = model.tokenizer.convert_ids_to_tokens(input_id)
        sentence = model.tokenizer.convert_tokens_to_string(tokens)
        sentences_recovered.append(sentence)
    # assert sentences_recovered == sentences_batch, f"Recovered sentences do not match original. Recovered: {sentences_recovered}, Original: {sentences_batch}"
    print(features)
    print(sentences_recovered)
```
改进后查看model.encode的中间步骤（包括tokenize的过程）输出结果，输出结果如下图（注意每一次输出都不一样，是因为batch-size设置成了2）：

![改进后查看model.encode的中间步骤（包括tokenize的过程）输出结果](https://raw.githubusercontent.com/Sylviara/1st_try/main/image/20260118213919211.png)

如果把batch_size设置成1，就是一条条数据了：

![](https://raw.githubusercontent.com/Sylviara/1st_try/main/image/20260118214148214.png)


```python

# 最终验证我的想法的代码：
print(model.encode_query(["我吃屎","我吃粪","我吃大便"], show_progress_bar=True,output_value="token_embeddings"))
print("==============================")
print(model.encode_query(np.array([["我吃屎"],["我吃粪"],["我吃大便"]]), show_progress_bar=True))
# 在这里，输入np array的话就会报错。
```



验证好了这一想法后，我的使用dataframe对原始数据id进行记录的梦想就破碎了。但是我们还是可以对数据id进行记录，比如下图：

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# 加载 Qwen-3-0.6B 模型
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# 训练数据
data = [
    ["头痛", "脑袋痛死"],
    ["头晕", "晕乎乎"],
    ["呕吐", "呕", "干呕"],
    ["头昏", "昏昏沉沉"]
]

# 为每个词汇生成唯一 ID
train_ids = []
flattened_data = []
for group_id, group in enumerate(data):
    for word in group:
        flattened_data.append(word)
        train_ids.append(f"train_{group_id}_{word}")

# 生成训练数据的嵌入
embeddings = model.encode(flattened_data, show_progress_bar=True)

# K-means 聚类
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 新数据示例
new_data = ["头痛", "头晕", "呕吐", "头昏", "眩晕", "脑袋痛死"]

# 为新数据生成唯一 ID
new_ids = [f"new_{i}_{word}" for i, word in enumerate(new_data)]

# 生成新数据的嵌入
new_embeddings = model.encode(new_data, show_progress_bar=True)

# 计算新数据与每个聚类中心的距离
distances = np.linalg.norm(new_embeddings[:, np.newaxis] - centroids, axis=2)

# 找到每个新数据点最近的聚类中心
closest_centroids = np.argmin(distances, axis=1)

# 找到靠近每个聚类中心的训练数据
num_neighbors = 5
for cluster_index in range(num_clusters):
    # 找到距离该聚类中心最近的训练数据点
    cluster_distances = np.linalg.norm(embeddings - centroids[cluster_index], axis=1)
    nearest_indices = np.argsort(cluster_distances)[:num_neighbors]
    
    print(f"聚类中心 {cluster_index} 附近的词汇及其原始 ID:")
    for idx in nearest_indices:
        print(f"词汇: {flattened_data[idx]}, 原始 ID: {train_ids[idx]}")

# 显示新数据的聚类标签及其原始 ID
for i, (word, label) in enumerate(zip(new_data, closest_centroids)):
    print(f"新词汇: {word}, 聚类标签: {label}, 原始 ID: {new_ids[i]}")

```

这样一来，就可以在输出结果的时候显示id了：

![](https://raw.githubusercontent.com/Sylviara/1st_try/main/image/20260118214530766.png)