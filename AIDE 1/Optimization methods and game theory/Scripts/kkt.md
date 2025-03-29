# KKT

Given the problem
$$\begin{cases}
  \min f(x) \\
  g_i(x) \leq 0\forall i\\
  h_i(x) = 0\forall i
\end{cases}\hspace{30pt}(P)$$

## Abadie Constraint Qualification

ACQ holds if $T_x(x^*)=D(x^*)$

### Sufficient conditions

- Affine constraints
- Slater's condition

  if all the $g_i$ are convex and all the $h_i$ are affine and there exists an interior point in the feasible set $\bar{x}:g_i(\bar{x})<0\forall i$ and $h_i(\bar{x})=0\forall i$

- Linear independence of the gradients of active constraints

## KKT System

One of the solutions to the KKT system is the optimal solution to the problem

$$\begin{cases}
  \nabla f(x^*)+\sum_{i=1}^m\lambda_i\nabla g_i(x^*)+\sum_{i=1}^p\mu_i\nabla h_i(x^*)=0\\
  g_i(x^*)\leq0\forall i\\
  h_i(x^*)=0\forall i\\
  \lambda_i\geq0\forall i\\
  \lambda_ig_i(x^*)=0\forall i
\end{cases}$$

If the problem is convex the solution is a global optimum.

The $\inf L(x,\lambda,\mu)$ is the Lagrangian relaxation of the problem P, provides a lower bound to the optimal value of P.

$\varphi(\lambda,\mu)=\inf_{x\in\mathbb{R}^n}L(x,\lambda,\mu)$ is the dual function.

This is the dual problem
$$\begin{cases}
  \max\varphi(\lambda,\mu)\\
  \lambda\geq0
\end{cases}\hspace{30pt}(D)$$

$v(D)\leq v(P)$

The dual problem is always a convex optimization problem even if the primal problem is not.

If the primal is continuously differentiable and convex and ACQ holds at the solution, then v(D)=v(P).

$L(x^*,\lambda,\mu)\leq L(x^*,\lambda^*,\mu^*)\leq L(x,\lambda^*,\mu^*)$ iff $x^*,\lambda^*,\mu^*$ is an optimal solution and strong duality holds.
