# Support Vector Machines

## Linear SVM

We must solve the problem
$$\begin{cases}
  \min\frac{1}{2}||w||^2\\
  1-y_i(w^Tx_i+b)\leq0\forall i
\end{cases}$$

The dual problem is
$$\begin{cases}
  \max_\lambda-\frac{1}{2}\sum_{i}\sum_{j}y^iy^j(x^i)^Tx^j\lambda_i\lambda_j+\sum_{i}\lambda_i\\
  \sum_{i}y^i\lambda_i=0\\
  \lambda_i\geq0\forall i
\end{cases}$$

Also written as
$$\begin{cases}
  \max_\lambda-\frac{1}{2}\lambda^TX^TX\lambda+1^T\lambda\\
  y^T\lambda=0\\
  \lambda\geq0
\end{cases}$$

Where $X=\{y^ix^i\}$.

## Linear SVM with slack variables

We must solve the problem
$$\begin{cases}
  \min\frac{1}{2}||w||^2+C\sum_{i}\xi_i\\
  1-y_i(w^Tx_i+b)\leq-\xi_i\forall i\\
  \xi_i\geq0\forall i
\end{cases}$$

The dual problem is
$$\begin{cases}
  \max_\lambda-\frac{1}{2}\sum_{i}\sum_{j}y^iy^j(x^i)^Tx^j\lambda_i\lambda_j+\sum_{i}\lambda_i\\
  \sum_{i}y^i\lambda_i=0\\
  0\leq\lambda_i\leq C\forall i
\end{cases}$$

Also written as
$$\begin{cases}
  \max_\lambda-\frac{1}{2}\lambda^TX^TX\lambda+1^T\lambda\\
  y^T\lambda=0\\
  0\leq\lambda\leq C
\end{cases}$$

Where $X=\{y^ix^i\}$.

## Kernel SVM with slack variables

We must solve the problem
$$\begin{cases}
  \min\frac{1}{2}||w||^2+C\sum_{i}\xi_i\\
  1-y_i(w^T\phi(x_i)+b)\leq-\xi_i\forall i\\
  \xi_i\geq0\forall i
\end{cases}$$

$w$ might be infinite-dimensional so we have to use the dual problem.

The dual problem is
$$\begin{cases}
  \max_\lambda-\frac{1}{2}\sum_{i}\sum_{j}y^iy^j\phi(x^i)^T\phi(x^j)\lambda_i\lambda_j+\sum_{i}\lambda_i\\
  \sum_{i}y^i\lambda_i=0\\
  0\leq\lambda_i\leq C\forall i
\end{cases}$$

Also written as
$$\begin{cases}
  \max_\lambda-\frac{1}{2}\lambda^TK\lambda+1^T\lambda\\
  y^T\lambda=0\\
  0\leq\lambda\leq C
\end{cases}$$

Where $K=\{k_{ij}=y^iy^jk(x^i,x^j)\}$ is the kernel matrix and $k(x^i,x^j)=\phi(x^i)^T\phi(x^j)$ is the kernel function.

In this way we never need to compute $\phi(x)$.

We then choose an $i$ s.t. $0<\lambda_i<C$ and compute $b^*=\frac{1}{y^i}-\sum_{j}\lambda^*_jy^jk(x^i,x^j)$.

The decision function will be $f(x)=\sum_{i}\lambda^*_iy^ik(x^i,x)+b^*$.
