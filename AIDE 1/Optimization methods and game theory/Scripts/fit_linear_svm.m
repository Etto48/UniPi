function [w,b] = fit_linear_svm(A,B)
    % Fit a linear SVM model to the data
    % Inputs:
    % A : dxn matrix with the points from class 1
    % B : dxm matrix with the points from class 2

    % Outputs:
    % w : dx1 vector with the weights
    % b : scalar with the bias
    
    nA = size(A,1);
    nB = size(B,1);
    T = [A; B];
    Q = [eye(size(T,2)) zeros(size(T,2),1); zeros(1,size(T,2)) 0];
    D = [-A -ones(nA,1); B ones(nB,1)];
    d = -ones(nA+nB,1);
    [x,~,exitflag,~,~] = quadprog(Q,zeros(size(T,2)+1,1),D,d);

    if exitflag ~= 1
        error("Error while fitting the SVM model");
    end

    w = x(1:end-1);
    b = x(end);

    
    % if 2d data
    if size(T,2) == 2
        % Plot the data and the decision boundary
        figure;
        hold on;
        x_min_max = [min(T(:,1)) max(T(:,1))];
        y_min_max = [min(T(:,2)) max(T(:,2))];
        x_axis = linspace(x_min_max(1),x_min_max(2),100);
        decision_boundary = (-w(1)/w(2)).*x_axis - b/w(2);
        margin_pos = (-w(1)/w(2)).*x_axis + (1-b)/w(2);
        margin_neg = (-w(1)/w(2)).*x_axis + (-1-b)/w(2);
        plot(...
            A(:,1),...
            A(:,2),...
            "bo",...
            ...
            B(:,1),...
            B(:,2),...
            "ro",...
            ...
            x_axis,...
            decision_boundary,...
            "k-",...
            ...
            x_axis,...
            margin_pos,...
            "b-",...
            ...
            x_axis,...
            margin_neg,...
            "r-",...
            ...
            "Linewidth", 1.5);
        axis([x_min_max y_min_max]);
        xlabel("x1");
        ylabel("x2");
        title("Linear SVM model");
        legend("Class 1","Class 2","Decision boundary","Positive margin","Negative margin");
        hold off;
    end
end