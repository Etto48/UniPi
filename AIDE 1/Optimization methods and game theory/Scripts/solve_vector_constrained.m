function [solutions] = solve_vector_constrained(f, x0, alpha_min, alpha_max, A, b, D, e, low, up, nonlcon)
    % Solve a vector constrained problem
    % Inputs:
    % f - a cell array of anonymous function handles, each @(x) -> scalar
    % x0 - initial guess
    % alpha_min - minimum value for alpha
    % alpha_max - maximum value for alpha
    % A - matrix of constraints
    % b - vector of constraint bounds
    % D - matrix of equality constraints
    % e - vector of equality constraint bounds
    % low - lower bounds for x
    % up - upper bounds for x
    % nonlcon - function handle for nonlinear constraints
    % Outputs:
    % solutions - a matrix of solutions
    % Example usage:
    % f1 = @(x) x(1)^2 + x(2)^2;
    % f2 = @(x) (x(1)-1)^2 + (x(2)+1)^2;
    % f = {f1, f2};
    % x0 = [0; 0];
    % alpha_min = 0;
    % alpha_max = 1;
    % A = [1, 2; -1, 1];
    % b = [4; 1];
    % D = [1, -1];
    % e = [0];
    % low = [-5; -5];
    % up = [5; 5];
    % nonlcon = @(x) deal(1 - (x(1)-1)^2 - (x(2)-1)^2, x(1)^2 + x(2)^2 - 2);
    % solutions = solve_vector_constrained(f, x0, alpha_min, alpha_max, A, b, D, e, low, up, nonlcon, nonlcon);

    if nargin < 11
        nonlcon = @(x) deal([], []);  % Default to no nonlinear constraints
    end
    if nargin < 10
        up = [];  % Default to no upper bounds
    end
    if nargin < 9
        low = [];  % Default to no lower bounds
    end
    if nargin < 8
        e = [];  % Default to no equality constraints
    end
    if nargin < 7
        D = [];  % Default to no equality constraint matrix
    end
    if nargin < 6
        b = [];  % Default to no inequality constraint bounds
    end
    if nargin < 5
        A = [];  % Default to no inequality constraint matrix
    end
    if nargin < 4
        alpha_max = 1;  % Default maximum value for alpha
    end
    if nargin < 3
        alpha_min = 0;  % Default minimum value for alpha
    end

    if ~iscell(f)
        error('Input f must be a cell array of anonymous function handles.');
    end

    function [v] = objective(x, alpha)
        v = 0;
        for j = 1:length(f)
            v = v + alpha(j) * f{j}(x);
        end
    end

    dim = length(f);
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
        x = fmincon(@(x) objective(x, alpha), x0, A, b, D, e, low, up, nonlcon);
        if isempty(solutions)
            solutions = x.';
        else
            solutions = [solutions; x.'];
        end
    end
    % Remove duplicates
    [~, unique_indices] = unique(solutions, 'rows');
    solutions = solutions(unique_indices, :);
end