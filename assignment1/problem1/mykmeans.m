function cluster = mykmeans(X, k, c)

    % Find Euclidean Distances given a coord and a centroid vector
    function dist = L2(p, cent)
        dist = zeros(size(cent,1),1);
        for L2ind = 1:size(cent,1)
            dist(L2ind) = sqrt((cent(L2ind,1)-p(1)).^2 + (cent(L2ind,2)-p(2)).^2);
        end
    end

    % Find Euclidean Distances between two matricies and return a vector
    function dist = L2centers(c1, c2)
        dist = zeros(size(c1,1),1);
        for L2ind = 1:size(c1,1)
            dist(L2ind) = sqrt((c1(L2ind,1)-c2(L2ind,1)).^2 + ...
                               (c1(L2ind,2)-c1(L2ind,2)).^2);
        end
    end

    % Create cluster array where: row = X index; value = c index
    cluster = zeros(size(X,1),1);
    
    % Keep track of iteration # and new centers    
    iteration = 0;
    old_c = c+1;
    
    while min(L2centers(c, old_c)) >= 0.001 && iteration <= 10000
        
        % Update iteration #
        iteration = iteration + 1;
        
        % Set closest clusters
        for i = 1:size(X,1)
            [~,cluster(i)] = min(L2( X(i,:), c ));
        end
        
        % Keep track of previous centers
        old_c = c;
        
        % Converge centers
        for i = 1:k
            data = X(cluster == i, :);
            
            newX = sum(data(:,1)) / size(data,1);
            newY = sum(data(:,2)) / size(data,1);
            
            c(i,:) = [newX, newY];
        end
        
        % DEBUG: Plot new centers and data
%         C = {'r','b','m','g'};
%         
%         hold on
%         for i = 1:4
%             A3 = find(cluster == i);
%             plot(X(A3,1), X(A3,2),'+','MarkerEdgeColor', C{i});
%         end
%         
%         plot(c(:,1),c(:,2),'O', 'MarkerSize', 25);
%         L2centers(c, old_c)
%         
%         close all
        
    end
    
    % Print iteration # and new centers
    iteration
    sortrows(c)

end

