function [alpha] = k_median(data,k)
    % Solve the k-median clustering problem
    % Inputs:
    % data - an nxd matrix of data points
    % k - the number of clusters
    % Outputs:
    % alpha - an nx1 vector of cluster assignments
    n = size(data,1);
    d = size(data,2);
    % Initialize the cluster centers
    centers = data(randperm(n,k),:);

    function [a] = dist(x,y)
        a = 0;
        for i = 1:d
            a = a + abs(x(i)-y(i));
        end
    end

    function [v] = objective()
        v = 0;
        for i = 1:n
            for j = 1:k
                v = v + alpha(i,j)*dist(data(i,:),centers(j,:));
            end
        end
    end

    function [A] = assign_clusters(P,X)
        A = zeros(n,k);
        for i = 1:n
            assignment = 1;
            for j = 1:k
                if dist(X(i,:),P(j,:)) < dist(X(i,:),P(assignment,:))
                    assignment = j;
                end
            end
            A(i,assignment) = 1;
        end
    end 

    function [X] = update_centers(A,P)
        X = zeros(k,d);
        for i = 1:k
            X(i,:) = median(P(A(:,i)==1,:),1);
        end
    end

    % Initialize the cluster assignments
    alpha = assign_clusters(centers,data);

    while true
        v = objective();
        % Update cluster centers
        centers = update_centers(alpha,data);
        % Update cluster assignments
        alpha = assign_clusters(centers,data);
        % Check for convergence
        if objective() == v
            break;
        end
    end

    % Plot the data and the cluster centers if 2d
    if d == 2
        figure;
        hold on;
        for idx = 1:k
            plot(data(alpha(:,idx)==1,1),data(alpha(:,idx)==1,2),'.','MarkerSize',10);
        end
        plot(centers(:,1),centers(:,2),'xk','MarkerSize',10);
        hold off;
    end
end

