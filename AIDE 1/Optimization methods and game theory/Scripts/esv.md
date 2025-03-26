# $\varepsilon$-SV Regression

## Linear $\varepsilon$-SV Regression

The problem is
$$\begin{cases}
    \text{min }\frac{1}{2}||w||^2\\
    |y_i-w^Tx_i-b|\leq\varepsilon\forall i
\end{cases}$$

We can split the absolute value into two inequalities:

$$\begin{cases}
    \text{min }\frac{1}{2}||w||^2\\
    y_i-w^Tx_i-b\leq\varepsilon\forall i\\
    -y_i+w^Tx_i+b\leq\varepsilon\forall i
\end{cases}$$

## Linear $\varepsilon$-SV Regression with slack variables

The problem is
$$\begin{cases}
    \text{min }\frac{1}{2}||w||^2+C\sum_{i}\xi_i\\
    y_i-w^Tx_i-b\leq\varepsilon+\xi^+_i\forall i\\
    -y_i+w^Tx_i+b\leq\varepsilon+\xi^-_i\forall i\\
    \xi^+_i\geq0\forall i\\
    \xi^-_i\geq0\forall i
\end{cases}$$

The dual problem is
$$\begin{cases}
    \text{max}_{\lambda^+,\lambda^-}-\frac{1}{2}\sum_{i}\sum_{j}(\lambda^+_i-\lambda^-_i)(\lambda^+_j-\lambda^-_j)(x^i)^Tx^j\\\hspace{50pt}-\varepsilon\sum_{i}(\lambda^+_i+\lambda^-_i)+\sum_{i}y_i(\lambda^+_i-\lambda^-_i)\\
    \sum_{i}(\lambda^+_i-\lambda^-_i)=0\\
    0\leq\lambda^+_i\leq C\forall i\\
    0\leq\lambda^-_i\leq C\forall i
\end{cases}$$

In matrix form
$$\begin{cases}
    \text{max}_{\lambda^+,\lambda^-}-\frac{1}{2}\begin{pmatrix}\lambda^+\\\lambda^-\end{pmatrix}^TQ\begin{pmatrix}\lambda^+\\\lambda^-\end{pmatrix}+\left(-\varepsilon 1^T+\begin{pmatrix}y\\-y\end{pmatrix}^T\right)\begin{pmatrix}\lambda^+\\\lambda^-\end{pmatrix}\\
    1^T(\lambda^+-\lambda^-)=0\\
    0\leq\lambda^+\leq C\\
    0\leq\lambda^-\leq C
\end{cases}$$

With $Q=\begin{pmatrix}K & -K\\-K & K\end{pmatrix}$ and $K_{ij}=x^i\cdot x^j$.

We can find $w$ and $b$ from the dual solution
$$w=\sum_{i}(\lambda^+_i-\lambda^-_i)x_i$$
If $\exists i$ s.t. $0<\lambda^+_i<C$ then $b=y_i-w^Tx_i-\varepsilon$.

If $\exists i$ s.t. $0<\lambda^-_i<C$ then $b=y_i-w^Tx_i+\varepsilon$.

## Non-linear $\varepsilon$-SV Regression

We can just replace $x^i\cdot x^j$ with $K(x^i,x^j)$ in the dual problem of the linear $\varepsilon$-SV Regression with slack variables.

The regression function is
$$f(x)=\sum_{i}(\lambda^+_i-\lambda^-_i)K(x^i,x)+b$$
Where $b$ is computed as

If $\exists i$ s.t. $0<\lambda^+_i<C$ then $$b=y_i-\varepsilon-\sum_{j}(\lambda^+_j-\lambda^-_j)K(x^j,x^i)$$

If $\exists i$ s.t. $0<\lambda^-_i<C$ then $$b=y_i+\varepsilon-\sum_{j}(\lambda^+_j-\lambda^-_j)K(x^j,x^i)$$
