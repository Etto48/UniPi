# Solution methods

## Linear equality constraints

The problem
$$\begin{cases}
\min f(x) \\
Ax = b
\end{cases}$$

$A$ can be written as $A = [A_1, A_2]$ with $det(A_1) \neq 0$ and $A_1\in\mathbb{R}^{p\times p}$.
$x$ can be written as $x = [x_1, x_2]$ with $x_1\in\mathbb{R}^p$

We can then set $x_1=A_1^{-1}(b-A_2x_2)$ and thus eliminating the variables $x_1$ from the problem.

The problem becomes unconstrained in $n-p$ variables $x_2$

$$\begin{cases}
\min f(A_1^{-1}(b-A_2x_2), x_2) \\
x_2\in\mathbb{R}^{n-p}
\end{cases}$$

## Penalty method

The problem
$$\begin{cases}
\min f(x) \\
g_i(x) \leq 0\forall i
\end{cases}$$

With $X$ the feasible set.

We can define the penalty function
$$p(x) = \sum_{i=1}^m \max(0, g_i(x))$$

And then the problem becomes
$$\begin{cases}
\min f(x) + \frac{1}{\varepsilon}p(x) := f_\varepsilon(x) \\
x\in\mathbb{R}^n
\end{cases}$$

If $x^*$ solves $(P_\varepsilon)$ and $x^*\in X$ then $x^*$ also solves $(P)$.

The algorithm to find the solution of $(P)$ is
1. Set $\varepsilon = \varepsilon_0$ and $\tau\in(0,1)$.
2. Solve $(P_\varepsilon)$ and get $x^*$.
3. Then
   - If $x^*\in X$ then STOP.
   - Else set $\varepsilon = \tau\varepsilon$ and go to step 2.

## Logarithmic barrier method

The problem
$$\begin{cases}
\min f(x) \\
g_i(x) \leq 0\forall i
\end{cases}$$
With $X$ the feasible set.

Can be approximated inside $\text{int}(X)$ by the problem
$$\begin{cases}
\min f(x) - \varepsilon\sum_{i=1}^m \log(-g_i(x)) := \psi_\varepsilon(x) \\
x\in\text{int}(X)
\end{cases}$$

We call $B(x)=-\sum_{i=1}^m \log(-g_i(x))$ the barrier function.
So $\psi_\varepsilon(x) = f(x) - \varepsilon B(x)$.

Note that as $x$ approaches the boundary of $X$, $\psi_\varepsilon(x)\rightarrow+\infty$

If $x^*$ is a local minimum of $(P_\varepsilon)$ the
$$\nabla\psi_\varepsilon(x^*) = \nabla f(x^*) - \varepsilon\sum_{i=1}^m \frac{\nabla g_i(x^*)}{-g_i(x^*)}=0$$

We can show that $v(P)=v(P_\varepsilon) - m\varepsilon$ where $m$ is the number of constraints.

The algorithm is
1. Set the tolerance $\delta>0$ and $\tau\in(0,1)$ and $\varepsilon_1>0$. Choose $x^0\in \text{int}X$ set $k=1$
2. Find the optimal solution $x^k$ of
    $$\begin{cases}
        \min \psi_\varepsilon(x)\\
        x\in\text{int}X
    \end{cases}$$
    using $x^{k-1}$ as a starting point.
3. Then
   - If $m\varepsilon_k<\delta$ then STOP.
   - Else $\varepsilon_{k+1}=\tau\varepsilon_k$ and $k=k+1$ and go to step 2.

To find a starting point $x^0$ we can solve the problem
$$\begin{cases}
    \min s \\
    g_i(x) \leq s\forall i
\end{cases}$$
With the Logarithmic barrier method starting from any $\tilde{x}\in\mathbb{R}^n$ and $\tilde{s}>\max g(\tilde{x})$.
If $s^*<0$ then $x^*\in\text{int}X$, otherwise $\text{int}X = \emptyset$.
