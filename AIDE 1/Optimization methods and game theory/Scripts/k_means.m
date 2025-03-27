function [alpha] = k_means(data,k)
    % Solve the k-means clustering problem
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
        a = (x-y)'*(x-y);
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
            X(i,:) = sum(A(:,i).*P,1)/sum(A(:,i));
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
        % Plot the Voronoi diagram
        [X,Y] = meshgrid(linspace(min(data(:,1)),max(data(:,1)),100),linspace(min(data(:,2)),max(data(:,2)),100));
        Z = zeros(100,100);
        for idx = 1:100
            for jdx = 1:100
                assignment = 1;
                for l = 1:k
                    if dist([X(idx,jdx),Y(idx,jdx)],centers(l,:)) < dist([X(idx,jdx),Y(idx,jdx)],centers(assignment,:))
                        assignment = l;
                    end
                end
                Z(idx,jdx) = assignment;
            end
        end
        contour(X,Y,Z);
        for idx = 1:k
            plot(data(alpha(:,idx)==1,1),data(alpha(:,idx)==1,2),'.','MarkerSize',10);
        end
        plot(centers(:,1),centers(:,2),'xk','MarkerSize',10);
        hold off;
    end
end

