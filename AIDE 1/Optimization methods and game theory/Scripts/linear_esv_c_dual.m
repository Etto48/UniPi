function [w,b] = linear_esv_c_dual(X,y,e,C)
    % Fit a linear epsilon-SV regression model to the data using the dual
    % problem
    % Inputs:
    % X : nxd matrix with the x values
    % y : nx1 vector with the y values
    % e : scalar with the width of the epsilon-tube
    % C : scalar error penalty
    % Outputs:
    % w : dx1 vector with the weights
    % b : dx1 vector with the bias

    K = X*X';
    n = size(X,1);
    Q = [K, -K; -K, K];
    f = e*ones(2*n,1) + [-y; y];
    Aeq = [ones(1,n), -ones(1,n)];
    beq = 0;
    lowerbound = zeros(2*n,1);
    upperbound = C*ones(2*n,1);
    sol = quadprog(Q, f, [], [], Aeq, beq, lowerbound, upperbound);
    la_plus = sol(1:n);
    la_minus = sol(n+1:end);
    w = (la_plus - la_minus)'*X;

    % Find the support vectors
    sv = find(la_plus > 1e-6 & la_plus < C - 1e-6);
    if ~isempty(sv)
        i = sv(1);
        b = y(i) - w'*X(i,:)' - e;
    else
        sv = find(la_minus > 1e-6 & la_minus < C - 1e-6);
        i = sv(1);
        b = y(i) - w'*X(i,:)' + e;
    end

    % Plot the data and the decision boundary if 2d
    if size(X,2) == 1
        figure;
        hold on;
        plot(X, y, "bo");
        x_axis = X;
        y_axis = w(1)*x_axis + b;
        plot(x_axis, y_axis, "k-");
        plot(x_axis, y_axis + e, "r-");
        plot(x_axis, y_axis - e, "r-");
        hold off;
    end
end