function [w,b] = linear_esv(X,y,e)
    % Fit a linear epsilon-SV regression model to the data
    % Inputs:
    % X : nxd matrix with the x values
    % y : nx1 vector with the y values
    % e : scalar with the width of the epsilon-tube
    % Outputs:
    % w : dx1 vector with the weights
    % b : dx1 vector with the bias

    Q = [eye(size(X,2)) zeros(size(X,2),1); zeros(1,size(X,2)) 0];
    D = [-X -ones(size(X,1),1); X ones(size(X,1),1)];
    d = e*ones(2*size(X,1),1) + [-y; y];
    f = zeros(size(X,2)+1,1);
    [x, ~, exitflag] = quadprog(Q,f,D,d);
    if exitflag ~= 1
        error("Error while fitting the epsilon-SV regression model");
    end
    w = x(1:end-1);
    b = x(end);

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