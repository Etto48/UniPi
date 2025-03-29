function [solutions] = solve_vector_constrained(f, x0, alpha_min, alpha_max, A, b, D, e, low, up, c, ceq)
    % Solve a vector constrained problem
    % Inputs:
    % f - a vector of objective functions
    % x0 - initial guess
    % alpha_min - minimum value for alpha
    % alpha_max - maximum value for alpha
    % A - matrix of constraints
    % b - vector of constraint bounds
    % D - matrix of equality constraints
    % e - vector of equality constraint bounds
    % low - lower bounds for x
    % up - upper bounds for x
    % c - function for nonlinear constraints
    % ceq - function for nonlinear equality constraints
    % Outputs:
    % solutions - a matrix of solutions

    function [v] = objective(x, alpha)
        v = 0;
        for j = 1:length(f)
            v = v + alpha(j) * f{j}(x);
        end
    end

    dim = length(x0);
    solutions = [];
    alpha_values = linspace(alpha_min, alpha_max, 10);  % 10-step linspace between a and b
    alpha_grids = cell(1, dim);
    [alpha_grids{:}] = ndgrid(alpha_values);   % Generate grid for all dimensions
    alpha_matrix = cell2mat(cellfun(@(x) x(:), alpha_grids, 'UniformOutput', false));  % Convert to matrix
    all_0s_index = all(alpha_matrix == 0, 2);  % Find all-zero rows
    alpha_matrix(all_0s_index, :) = [];  % Remove all-zero rows
    
    n_iter = size(alpha_matrix, 1);
    for i = 1:n_iter
        alpha = alpha_matrix(i, :);
        x = fmincon(@(x) objective(x, alpha), x0, A, b, D, e, low, up, c, ceq);
        if isempty(solutions)
            solutions = x;
        else
            solutions = [solutions; x];
        end
    end
    % Remove duplicates
    [~, unique_indices] = unique(solutions, 'rows');
    solutions = solutions(unique_indices, :);
end
    