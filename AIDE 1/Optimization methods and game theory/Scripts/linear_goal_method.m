function [x] = linear_goal_method(C,A,b)
    % Solve a vector linear constrained problem using the goal
    % method
    % Inputs:
    % C - a matrix of coefficients
    % A - a matrix of constraints
    % b - a vector of constraint bounds
    % Outputs:
    % x - the solution vector

    s = size(C, 1);
    z = zeros(s, 1);

    for i = 1:s
        [~,z(i)] = linprog(C(i,:)', A, b);
    end

    H = C'*C;
    f = -C'*z;
    x = quadprog(H, f, A, b);
end