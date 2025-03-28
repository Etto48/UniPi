function [x] = solve_quadratic_barrier(Q, c, A, b, x0, d, e0, t)
    % Solve a quadratic constrained problem using the logarithmic barrier method
    % Inputs:
    % Q - a positive definite matrix
    % c - a vector of coefficients
    % A - a matrix of constraints
    % b - a vector of constraint bounds
    % x0 - an initial guess
    % d - tolerance for convergence
    % e0 - initial barrier coefficient
    % t - barrier increment
    % Outputs:
    % x - the solution vector
    
    e = e0;
    x = x0;
    m = size(A, 1);

    function [v] = f(x)
        v = 0.5 * x' * Q * x + c' * x;
    end

    function [gv] = g(x)
        gv = A * x - b;
    end

    function [barrier] = B(x)
        barrier = -sum(log(-g(x)));
    end
    
    function [v] = objective(x)
        v = f(x) + e * B(x);
    end

    while true
        x = fminunc(@objective, x);
        if m*e < d
            break;
        end
        e = e * t;
    end
end