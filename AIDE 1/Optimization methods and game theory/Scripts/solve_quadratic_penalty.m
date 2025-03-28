function [x] = solve_quadratic_penalty(Q, c, A, b, x0, e0, t)
    % Solve a quadratic constrained problem using the quadratic penalty method
    % Inputs:
    % Q - a positive definite matrix
    % c - a vector of coefficients
    % A - a matrix of constraints
    % b - a vector of constraint bounds
    % x0 - an initial guess
    % e0 - initial penalty coefficient
    % t - penalty increment
    % Outputs:
    % x - the solution vector
    
    e = e0;
    x = x0;

    function [v] = f(x)
        v = 0.5 * x' * Q * x + c' * x;
    end

    function [gv] = g(x)
        gv = A * x - b;
    end

    function [v] = objective(x)
        v = f(x) + 1/e * sum(max(0, g(x)).^2);
    end

    while true
        x = fminunc(@objective, x);
        g_val = g(x);
        if max(g_val) < 1e-6
            break;
        end
        e = e * t;
    end
end