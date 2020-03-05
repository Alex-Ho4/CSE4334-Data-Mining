restoredefaultpath;
clear all;
close all;

%% Part 2

R2 = normrnd(5,1,1,1000);

band = [.1, 1, 5, 10];

for h = 1:size(band,2)
    figure(h);
    hold on
    hist = histogram(R2, 'BinWidth', band(h), 'Normalization', 'probability');
    [p, x] = mykde(R2, band(h));    
    
    plot(x, p, 'g','LineWidth', 2);
    hold off
    
    % DEBUG: Compare to built in ksdensity
    % figure(4+h);
    % hold on
    % hist = histogram(R2, 'BinWidth', band(h), 'Normalization', 'probability');
    % [f, xi] = ksdensity(R2,'Bandwidth',band(h));
    %     
    % plot(xi, f);
    % hold off
end

%% Part 3

R3 = normrnd(0,0.2,1,1000);

band = [.1, 1, 5, 10];

for h = 1:size(band,2)
    figure(4+h);
    hold on
    hist = histogram(R3, 'BinWidth', band(h), 'Normalization', 'probability');
    [p, x] = mykde(R3, band(h));    
    
    plot(x, p, 'g','LineWidth', 2);
    hold off
    
%     figure(4+h);
%     hold on
%     hist = histogram(R3, 'BinWidth', band(h), 'Normalization', 'probability');
%     [f, xi] = ksdensity(R3,'Bandwidth',band(h));
%         
%     plot(xi, f);
%     hold off
end

%% Part 4

mu1 = [1,0];
mu2 = [0,1.5];

sigma = [0.9, 0.4 ; 0.4, 0.9];

N1 = mvnrnd(mu1, sigma, 500);
N2 = mvnrnd(mu2, sigma, 500);

for h = 1:size(band,2)
    % Get kde of N1 with bandwidth band(h)
    [p, x] = mykde(N1, band(h));

    % Plot N1
    figure(8+h);
    x1 = linspace(min(x(:,1)),max(x(:,1)));
    y2 = linspace(min(x(:,2)),max(x(:,2)));
    [xq, yq] = meshgrid(x1,y2);
    z = griddata(x(:,1),x(:,2),p,xq,yq);
    surf(xq, yq, z);
end

for h = 1:size(band,2)
    % Get kde of N2 with bandwidth band(h)
    [p, x] = mykde(N2, band(h));

    % Plot N2
    figure(12+h);
    x1 = linspace(min(x(:,1)),max(x(:,1)));
    y2 = linspace(min(x(:,2)),max(x(:,2)));
    [xq, yq] = meshgrid(x1,y2);
    z = griddata(x(:,1),x(:,2),p,xq,yq);
    surf(xq, yq, z);
end


