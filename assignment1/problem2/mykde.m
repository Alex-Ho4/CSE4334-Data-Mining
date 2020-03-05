function [p, x] = mykde(X, h)
    
    % For 1D array
    if(size(X,1) == 1)
        
        p = zeros(size(X));
        x = zeros(size(X));
        
        % Sort so it can be plotted
        X = sort(X);
        
        n = numel(X);
        
        for i = 1:n
           % Set domain of x
           x(i) = X(i);
           
           % u = distance / h
           u = (X - X(i)) / h;
           % Set k to 1 if u <= .5, otherwise 0
           k = abs(u) <= .5;
           
           % k = Gaussian Kernel Function
           % k = (1 / sqrt(2 * pi)) * exp( (-1/2) * u.^2 );
           
           p(i) = (1/(n*h)) * sum(k);
           
        end
        
    % For 2D matrix
    else
        
        p = zeros(size(X, 1), 1);
        x = zeros(size(X));
        
        % Sort so it can be plotted
        X = sortrows(X);
        
        [m, n] = size(X);

        % Utilized built in multivariate kernel density for probability
        x = X;
        F = mvksdensity(X,x,'BandWidth',h);
        p = F;
        
    end

end