function [la_plus,la_minus] = gaussian_esv(X,y,e,C,gamma)
    % Fit a gaussian epsilon-SV regression model to the data
    % Inputs:
    % X : nxd matrix with the x values
    % y : nx1 vector with the y values
    % e : scalar with the width of the epsilon-tube
    % C : scalar error penalty
    % gamma : scalar with the width of the gaussian kernel
    % Outputs:
    % la_plus : nx1 vector with the positive lagrange multipliers
    % la_minus : nx1 vector with the negative lagrange multipliers

    % function v = kernel(x,y)
    %     p = 4 ;
    %     v = (x'*y + 1)^p;
    % end

    function K = kernel(x1,x2)
        K = exp(-gamma*norm(x1-x2)^2);
    end

    n = size(X,1);
    K = zeros(n,n);
    for i = 1:n
        for j = 1:n
            K(i,j) = kernel(X(i),X(j));
        end
    end
    
    Q = [K -K; -K K];
    f = e*ones(2*n,1) + [-y; y];
    Aeq = [ones(1,n) -ones(1,n)];
    beq = 0;
    lowerbound = zeros(2*n,1);
    upperbound = C*ones(2*n,1);
    sol = quadprog(Q, f, [], [], Aeq, beq, lowerbound, upperbound);
    la_plus = sol(1:n);
    la_minus = sol(n+1:2*n);

    % Find the support vectors
    sv = find(la_plus > 1e-6 & la_plus < C - 1e-6);
    if ~isempty(sv)
        i = sv(1);
        b = y(i) - e;
        for j = 1:n
            b = b - (la_plus(j) - la_minus(j))*kernel(X(j,:),X(i,:));
        end
    else
        sv = find(la_minus > 1e-6 & la_minus < C - 1e-6);
        if isempty(sv)
            error("No support vectors found");
        end
        i = sv(1);
        b = y(i) + e;
        for j = 1:n
            b = b - (la_plus(j) - la_minus(j))*kernel(X(j,:),X(i,:));
        end
    end

    % Plot the data and the decision boundary if 2d
    if size(X,2) == 1
        figure;
        hold on;
        plot(X, y, "b.");
        x_axis = X;
        y_axis = zeros(n,1);
        for i = 1:n
            y_axis(i) = b;
            for j = 1:n
                y_axis(i) = y_axis(i) + (la_plus(j) - la_minus(j))*kernel(X(j,:),X(i,:));
            end
        end
        %plot sv
        plot(X(sv),y(sv),"ro");
        plot(x_axis, y_axis, "k-");
        plot(x_axis, y_axis + e, "r-");
        plot(x_axis, y_axis - e, "r-");
        hold off;
    end
end