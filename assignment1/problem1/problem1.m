restoredefaultpath;
clear all;
close all;

%% Part 1

mu1 = [1,0];
mu2 = [0,1.5];

sigma = [0.9, 0.4 ; 0.4, 0.9];

R1 = mvnrnd(mu1, sigma, 500);
R2 = mvnrnd(mu2, sigma, 500);

c1 = [10, 10 ;
      -10, -10];
  
c2 = [10, 10 ;
      -10, -10 ;
      10, -10 ;
      -10, 10 ];
  
C = {'r','b','m','g'};

%% Part 2

%p2 = mykmeans(R1, 2, c1);
% p2 = mykmeans(R1, 2, c1);
% p2 = mykmeans(R1, 2, c1);
% p2 = mykmeans(R1, 2, c1);
% p2 = mykmeans(R1, 2, c1);

% figure(1);
% hold on
% for i = 1:2
%     A2 = find(p2 == i);
%     plot(R1(A2,1), R1(A2,2),'+','MarkerEdgeColor', C{i});
% end
% hold off

% DEBUG: Compare to built in kmeans

% p2k = kmeans(R1, 2);
% 
% figure(3);
% hold on
% for i = 1:2
%     A2 = find(p2k == i);
%     plot(R1(A2,1), R1(A2,2),'+','MarkerEdgeColor', C{i});
% end
% hold off

%% Part 3

p3 = mykmeans(R2, 4, c2);
% p3 = mykmeans(R2, 4, c2);
% p3 = mykmeans(R2, 4, c2);
% p3 = mykmeans(R2, 4, c2);
% p3 = mykmeans(R2, 4, c2);

figure(2);
hold on
for i = 1:4
    A3 = find(p3 == i);
    plot(R2(A3,1), R2(A3,2),'+','MarkerEdgeColor', C{i});
end
hold off

% DEBUG: Compare to built in kmeans

[p3k, cD] = kmeans(R2, 4);

figure(4);
hold on
for i = 1:4
    A3 = find(p3k == i);
    plot(R2(A3,1), R2(A3,2),'+','MarkerEdgeColor', C{i});
end
hold off

sortrows(cD)
  
