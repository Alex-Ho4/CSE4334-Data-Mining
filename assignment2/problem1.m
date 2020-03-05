restoredefaultpath;
clear all;
close all;

%% Part 1

mu1 = [1,0];
mu2 = [0,1];

sigma = [1, 0.75 ; 0.75, 1];

C1 = mvnrnd(mu1, sigma, 500);
C2 = mvnrnd(mu2, sigma, 500);

L1 = zeros(size(C1,1), 1);
L2 = ones(size(C2,1), 1);

training = [C1;C2];
labels = [L1;L2];

%% Part 2

T1 = mvnrnd(mu1, sigma, 500);
T2 = mvnrnd(mu2, sigma, 500);
test = [T1; T2];
t_label = [zeros(size(T1,1), 1);ones(size(T2,1), 1)];

[pred, posterior, err] = myNB(training, labels, test, t_label);

accuracy = 1 - err;

% Confusion matrix / stats for class 0
cm_0 = [numel(find(pred == 0 & pred == t_label)), ...
       numel(find(pred == 0 & pred ~= t_label)); ...
       numel(find(pred == 1 & pred ~= t_label)), ...
       numel(find(pred == 1 & pred == t_label))];

recall_0 = cm_0(1,1) / (cm_0(1,1) + cm_0(2,1));
precision_0 = cm_0(1,1) / (cm_0(1,1) + cm_0(1,2));

% Confusion matrix / stats for class 1
cm_1 = [numel(find(pred == 1 & pred == t_label)), ...
        numel(find(pred == 1 & pred ~= t_label)); ...
        numel(find(pred == 0 & pred ~= t_label)), ...
        numel(find(pred == 0 & pred == t_label))];
   
recall_1 = cm_1(1,1) / (cm_1(1,1) + cm_1(2,1));
precision_1 = cm_1(1,1) / (cm_1(1,1) + cm_1(1,2));

class_0 = test(pred == 0, :);
class_1 = test(pred == 1, :);

figure(1);
hold on
scatter(class_0(:,1) , class_0(:,2), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
scatter(class_1(:,1) , class_1(:,2), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b');
hold off

%% Part 3

accs = zeros(1, 6);
k = 1;

for i = [10 20 50 100 300 500]

    T1_p3 = mvnrnd(mu1, sigma, i);
    T2_p3 = mvnrnd(mu2, sigma, i);
    
    test_p3 = [T1_p3; T2_p3];
    t_label_3 = [zeros(size(T1_p3,1), 1);ones(size(T2_p3,1), 1)];

    [~, ~, err3] = myNB(training, labels, test_p3, t_label_3);
    
    accs(1, k) = 1 - err3;
    k = k + 1;
    
end

figure(2);
plot([10 20 50 100 300 500], accs);

%% Part 4

C1_4 = mvnrnd(mu1, sigma, 700);
C2_4 = mvnrnd(mu2, sigma, 300);

L1_4 = zeros(size(C1_4,1), 1);
L2_4 = ones(size(C2_4,1), 1);

training_4 = [C1_4;C2_4];
labels_4 = [L1_4;L2_4];

[pred_4, post_4, err4] = myNB(training_4, labels_4, test, t_label);
    
acc_p4 = 1 - err4;

%% Part 5

confidence = posterior;
confidence(pred == 0) = 1-posterior(pred == 0);

TPR = zeros(20,1);
FPR = zeros(20,1);

k = 1;
for i = 1:-.05:0
    
    TP = numel(find(t_label(find(confidence >= i)) == 1));
    FN = numel(find(t_label(find(confidence < i)) == 1));
    
    FP = numel(find(t_label(find(confidence >= i)) == 0));
    TN = numel(find(t_label(find(confidence < i)) == 0));
    
    TPR(k) =  TP / (TP+FN);
    FPR(k) = FP / (TN+FP);
    
    k = k + 1;
    
end

figure(3);
plot(FPR, TPR);

AUC1 = trapz(.05, TPR)


confidence = post_4;
confidence(pred_4 == 0) = 1-post_4(pred_4 == 0);

TPR = zeros(20,1);
FPR = zeros(20,1);

k = 1;
for i = 1:-.05:0
    
    TP = numel(find(t_label(find(confidence >= i)) == 1));
    FN = numel(find(t_label(find(confidence < i)) == 1));
    
    FP = numel(find(t_label(find(confidence >= i)) == 0));
    TN = numel(find(t_label(find(confidence < i)) == 0));
    
    TPR(k) =  TP / (TP+FN);
    FPR(k) = FP / (TN+FP);
    
    k = k + 1;
    
end

figure(4);
plot(FPR, TPR);


AUC2 = trapz(.05, TPR)



