function [pred, posterior, err] = myNB(train_data,train_labels,test_data,test_labels)

train0 = train_data(train_labels == 0,:);
train1 = train_data(train_labels == 1,:);

prior0 = numel(find(train_labels == 0)) / size(train_data, 1);
prior1 = numel(find(train_labels == 1)) / size(train_data, 1);

mean0 = mean(train0);
std0 = std(train0);

mean1 = mean(train1);
std1 = std(train1);

pdf0 = normpdf(test_data, mean0, std0);
pdf1 = normpdf(test_data, mean1, std1);

t_pdf0 = pdf0(:,1) .* pdf0(:,2);
t_pdf1 = pdf1(:,1) .* pdf1(:,2);

post0 = (t_pdf0 .* prior0) ./ (t_pdf0.*prior0 + t_pdf1.*prior1);
post1 = (t_pdf1 .* prior1) ./ (t_pdf1.*prior1 + t_pdf0.*prior0);

pred = post1 > post0;
posterior = max(post0, post1);
err = numel(find(pred ~= test_labels)) / size(train_data, 1);

end

