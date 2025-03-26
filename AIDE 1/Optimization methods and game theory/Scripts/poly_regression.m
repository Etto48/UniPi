function [z] = poly_regression(x, y, n, norm)
    % Fit a polynomial regression model to the data
    % Inputs:
    % x : nx1 vector with the x values
    % y : nx1 vector with the y values
    % n : scalar with the degree of the polynomial
    % norm : 1 2 or Inf for the norm to use
    % Outputs:
    % z : (n+1)x1 vector with the weights

    % Create the Vandermonde matrix
    X = zeros(length(x), n+1);
    last_col = ones(length(x), 1);
    for j = 1:n+1
        X(:, j) = last_col;
        last_col = last_col.*x;
    end
    
    if norm == 2
        z = (X'*X)\(X'*y);
    elseif norm == 1
        c = [zeros(n+1, 1); ones(length(x), 1)];
        D = [X -eye(length(x)); -X -eye(length(x))];
        d = [y; -y];
        sol = linprog(c, D, d);
        z = sol(1:n+1);
    elseif norm == Inf
        c = [zeros(n+1, 1); 1];
        D = [X -ones(length(x),1); -X -ones(length(x),1)];
        d = [y; -y];
        sol = linprog(c, D, d);
        z = sol(1:n+1);
    else
        error("Invalid norm");
    end

    % Plot the data and the polynomial
    figure;
    hold on;
    plot(x, y, "bo");
    x_axis = x;
    y_axis = X*z;
    
    plot(x_axis, y_axis, "r-");
    hold off;
end