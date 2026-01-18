## 1. Question start：

when doing contrastive learning between image and text, one has to project all the embeddings of images and texts into one length space and calculate their dot-product, but while calculating they usually prefer to transpose one another. For example following is the contrastive loss calculation from [CLIP training script](https://colab.research.google.com/drive/1hYHb0FTdKQCXZs3qCwVZnSuVGrZU2Z1w?usp=sharing#scrollTo=7MQnmwsWi6lc). This is confusing to calculate the similarity with himself@himself.Transpose and confusing when calculate the total loss with just average of the two loss.

![来自CLIP](https://cdn.jsdelivr.net/gh/sylviara/sylviara.github.io@master/img/20240410145554.png)

## 2. What @ means in Python:

![dotproduct excercise from my understanding](https://cdn.jsdelivr.net/gh/sylviara/sylviara.github.io@master/img/20240410-dotproduct.png)

[矩阵相乘的一个解释](https://www.ruanyifeng.com/blog/2015/09/matrix-multiplication.html)



## 

