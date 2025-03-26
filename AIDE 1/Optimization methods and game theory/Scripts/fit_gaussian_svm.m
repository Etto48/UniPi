function [la] = fit_gaussian_svm(A,B,C,gamma)
    % Fit a Gaussian SVM model to the data
    % Inputs:
    % A : dxn matrix with the points from class 1
    % B : dxm matrix with the points from class 2
    % C : scalar error penalty
    % gamma : gaussian scale parameter
    % Outputs:
    % la : nx1 vector with the weights

    nA = size(A,1);
    nB = size(B,1);
    T = [A; B];
    % Kernel matrix
    K = exp(-gamma*pdist2(T,T).^2);
    y = [ones(nA,1); -ones(nB,1)];
    Q = (y*y').*K;
    f = -ones(nA+nB,1);
    Aeq = y';
    beq = 0;
    lowerbound = zeros(nA+nB,1);
    upperbound = C*ones(nA+nB,1);
    la = quadprog(Q,f,[],[],Aeq,beq,lowerbound,upperbound);

    % Plot the data if 2d
    if size(T,2) == 2
        figure;
        hold on;
        x_min_max = [min(T(:,1)) max(T(:,1))];
        y_min_max = [min(T(:,2)) max(T(:,2))];
        x_axis = linspace(x_min_max(1),x_min_max(2),100);
        y_axis = linspace(y_min_max(1),y_min_max(2),100);
        [X,Y] = meshgrid(x_axis,y_axis);
        % Z holds the value of the decision function at each point in the meshgrid
        Z = zeros(size(X));
        for i = 1:size(X,1)
            for j = 1:size(X,2)
                Z(i,j) = sum(la.*(y.*exp(-gamma*((X(i,j)-T(:,1)).^2 + (Y(i,j)-T(:,2)).^2))));
            end
        end
        contour(X,Y,Z,[0 0],'k-');
        plot(...
            A(:,1),...
            A(:,2),...
            "bo",...
            ...
            B(:,1),...
            B(:,2),...
            "ro",...
            ...
            "Linewidth", 1.5);
        axis([x_min_max y_min_max]);
    end
end