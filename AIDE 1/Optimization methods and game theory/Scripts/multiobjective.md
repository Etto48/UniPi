# Multiobjective optimization

The problem is defined as follows:
$$\begin{cases}
\min f(x) = (f_1(x), f_2(x), \ldots, f_s(x)) \\
x \in X
\end{cases}$$

Given $x,y\in\mathbb{R}^s$ we say that $x\geq y$ if $x_i\geq y_i \forall i=1,\ldots,s$.

- A point $\bar{x}$ is said to be Pareto ideal minimum if $\bar{x}\leq x \forall x\in X$
- A point $\bar{x}$ is said to be Pareto minimum if $\not\exists x\in X : x\neq\bar{x} $ and $\bar{x}\geq x$
- A point $\bar{x}$ is said to be Pareto weak minimum if $\not\exists x\in X : \bar{x}>x$ and $\bar{x}_i>x_i\forall i$

## Auxiliary optimization problem

$x^*\in X$ is a minimum of (P) iff the auxiliary optimization problem
$$\begin{cases}
    \max\sum_{i=1}^s \varepsilon_i \\
    f_i(x) + \varepsilon_i \leq f_i(x^*) \forall i\\
    x\in X\\
    \varepsilon\geq 0
\end{cases}$$
has optimal value $0$

$x^*\in X$ is a weak minimum of (P) iff the auxiliary optimization problem
$$\begin{cases}
    \max v \\
    v\leq \varepsilon_i \forall i\\
    f_i(x) + \varepsilon_i \leq f_i(x^*) \forall i\\
    x\in X\\
    \varepsilon\geq 0\\
    \sum_{i=1}^s \varepsilon_i = 0
\end{cases}$$
has optimal value $0$

If $x^*$ is a weak minimum then there exists $\theta^*\in\mathbb{R}^s$ such that $(x^*,\theta^*)$ is a solution of the system
$$\begin{cases}
    \sum_{i=1}^s \theta_i \nabla f_i(x) = 0 \\
    \theta_i\geq 0 \forall i\\
    \sum_{i=1}^s \theta_i = 1\\
    x\in\mathbb{R}^n\\
\end{cases}\hspace{30pt}(S)$$
If the problem is convex, the above condition is also sufficient. If $\theta^*>0$ then $x^*$ is a minimum.

## KKT

If $x^*$ is a weak minimum of (P) and ACQ holds at $x^*$,
then there exists $\theta^*\in\mathbb{R}^s,\lambda^*\in\mathbb{R}^m,\mu^*\in\mathbb{R}^p$ such that $(x^*,\theta^*,\lambda^*,\mu^*)$ is a solution of the system
$$\begin{cases}
    \sum_{i=1}^s \theta_i \nabla f_i(x) + \sum_{j=1}^m \lambda_j \nabla g_j(x) + \sum_{k=1}^p \mu_k \nabla h_k(x) = 0 \\
    \theta_i\geq 0 \forall i\\
    \sum_{i=1}^s \theta_i = 1\\
    \lambda \geq 0 \\
    \lambda_j g_j(x^*) = 0 \forall j\\
    g_j(x)\leq 0, h_k(x)=0
\end{cases}$$

If the problem is unconstrained then the KKT system reduces to $(S)$
If $\theta^*>0$ then $x^*$ is a minimum.

## Weighted sum method

Given the problem
$$\begin{cases}
    \min f(x) = (f_1(x), f_2(x), \ldots, f_s(x)) \\
    x\in X
\end{cases}\hspace{30pt}(P)$$
And a set of weights $\alpha = \{\alpha_1,\dots\alpha_s\}\geq 0$ associated with the objectives $f_i$.

We associate with $(P)$ the scalar problem
$$\begin{cases}
    \min \sum_{i=1}^s \alpha_i f_i(x) \\
    x\in X
\end{cases}\hspace{30pt}(P_\alpha)$$

The solutions of $(P_\alpha)$ are weak minima of $(P)$ if $\alpha_i\geq 0$ for all $i$ and are minima if $\alpha_i>0$ for all $i$.

If the problem is convex, any weak minimum of $(P)$ can be obtained given the right weights $\alpha$.

If $(P)$ is linear and $X$ is a polyhedron, then any minimum of $(P)$ can be obtained given the right weights $\alpha$.

## Goal method

Define $z_i = \min_{x\in X} f_i(x) \forall i$

We want to find the closest point to $z$ in $f(X)$.
$$\begin{cases}
    \min \sum_{i=1}^s ||f_i(x) - z_i||_q \\
    x\in X
\end{cases}\hspace{30pt}(G)$$

If $q\in[1,+\infty)$ then any optimal solution of $(G)$ is a minimum of $(P)$.

If $q=+\infty$ then any optimal solution of $(G)$ is a weak minimum of $(P)$.
