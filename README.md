# Probabilistic k-nearest nerighbours

![moons-pknn](https://imgur.com/8HYRSkZ.png)

Let  <img src="https://render.githubusercontent.com/render/math?math=\mathcal D = \big\{(y_n, {\bf x}_n) \vert y_n \in \{c_q\}_{q=1}^Q, {\bf x}_n\in\mathbb{R}^M\big\}"> be a dataset.

A pknn model attempts to find the posterior distribution of neighbours <img src="https://render.githubusercontent.com/render/math?math=p(\beta, k \vert \mathcal{D})"> given the likelihood function

<img src="https://render.githubusercontent.com/render/math?math=p(y_i \vert {\bf x}, \beta, k) = \frac{\exp\left(\frac{\beta}{k}\sum_{n\vert {\bf x}_n \in \mathcal{N}({\bf x}_i)}\mathbb{1}(y_n = y_i)\right)}{\sum_{q=1}^Q\exp\left(\frac{\beta}{k}\sum_{n\vert {\bf x}_n \in \mathcal{N}({\bf x}_i)}\mathbb{1}(y_n = q)\right)}">
