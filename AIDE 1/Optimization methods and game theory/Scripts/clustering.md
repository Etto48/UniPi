# Clustering

## K-means

The problem is
$$\begin{cases}
    \text{min }\sum_{i}\text{min}_{j=1...k}||p_i-x_j||^2_2\\
    x_j\in\mathbb{R}^n\forall j
\end{cases}$$

If $k=1$ the solution is $x_1=\frac{1}{n}\sum_{i}p_i$.

If $k>1$ the problem is non-convex and non differentiable.

If we fix $p_i$ and $x_j$ then the problem is

$$\begin{cases}
    \text{min }\sum_{j=1...k}\alpha_{ij}||p_i-x_j||^2_2\\
    \sum_{j=1...k}\alpha_{ij}=1\\
    \alpha_{ij}\geq0\forall j
\end{cases}$$

An optimal solution is $\alpha_{ij}=1$ if $j=\text{argmin}_{j=1...k}||p_i-x_j||^2_2$ and $0$ otherwise.

The problem is equivalent to
$$\begin{cases}
    \text{min }\sum_{i}\sum_{j=1...k}\alpha_{ij}||p_i-x_j||^2_2\\
    \sum_{j=1...k}\alpha_{ij}=1\\
    \alpha_{ij}\geq0\forall j\\
    x_j\in\mathbb{R}^n\forall j
\end{cases}$$

If we fix $x_j$ then the problem is decomposable in $n$ simple LP problems (in $\alpha_{ij}$).

$$\alpha^*_{ij}=\begin{cases}
    1 & \text{if }j=\text{argmin}_{j=1...k}||p_i-x_j||^2_2\\
    0 & \text{otherwise}
\end{cases}$$

If we fix $\alpha_{ij}$ then the problem is decomposable in $k$ convex QP problems in $x.

$$x^*_j=\frac{\sum_{i}\alpha_{ij}p_i}{\sum_{i}\alpha_{ij}}$$

We can then create an algorithm of alternating minimization:
1. Initialize $x_j$ randomly and assign $\alpha_{ij}$ as above.
2. Update $x_j$ as above.
3. Update $\alpha_{ij}$ as above.
4. Given $f(x,\alpha)=\sum_{i}\sum_{j}\alpha_{ij}||p_i-x_j||^2_2$
   - If $f(x^{t+1},\alpha^{t+1})=f(x^{t},\alpha^{t})$ then STOP.
   - Else go to Step 2.

## K-median

If we replace the $L_2$ norm with the $L_1$ norm we get the K-median problem.

The solution is equivalent to the K-means problem but we update $x_j$ with the median instead of the mean.
