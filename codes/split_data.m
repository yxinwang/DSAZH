function [XTrain,LTrain,XTest,LTest,XVal,LVal,class_unseen]=split_data(features,labels,dsname)
switch(dsname)
    case 'AWA1'
        c = 50; uc = 10; trn = 80; ten = 80;
    case 'AWA2'
        c = 50; uc = 10; trn = 100; ten = 100;
    case 'CUB'
        c = 200; uc = 50; trn = 30; ten = 30;
    case 'CIFAR10'
%         c = 10; uc = 2; trn = 3000; ten = 3000;
        c = 10; uc = 2; trn = 500; ten = 100;
    case 'aPY'
%         c = 32; uc = 6; trn = 50; ten = 50;
%         c = 32; uc = 8; trn = 50; ten = 30;
        c = 32; uc = 8; trn = 50; ten = 50;
    case 'SUN'
        %20 images per class
        c = 717; uc = 72; trn = 10; ten = 10;
end
class_unseen = randperm(c,uc);
class_seen = setdiff(1:c,class_unseen);

all_loc = 1:size(features,2);all_loc = all_loc';
all_loc_tag = zeros(size(features,2),1);

for i = 1:length(class_seen)
    temp_loc = all_loc(labels == class_seen(i));
    train_loc_i = randperm(length(temp_loc),trn);
    all_loc_tag(temp_loc(train_loc_i)) = 1;
end

for i = 1:length(class_unseen)
    temp_loc = all_loc(labels == class_unseen(i));
    test_loc_i = randperm(length(temp_loc),ten);
    all_loc_tag(temp_loc(test_loc_i)) = 2;
end

train_loc = all_loc(all_loc_tag == 1);
test_loc = all_loc(all_loc_tag == 2);
val_loc = all_loc(all_loc_tag == 0);

features = features';
% LAll = sparse(1:length(labels), double(labels), 1);
% revised in 2022/7/31 for adding full() function
LAll = full(sparse(1:length(labels), double(labels), 1));
XTrain = features(train_loc,:); LTrain = full(LAll(train_loc,:));
XTest = features(test_loc,:); LTest = full(LAll(test_loc,:));
XVal = features(val_loc,:); LVal = full(LAll(val_loc,:));

class_train_loc = unique(labels(train_loc));
class_test_loc = unique(labels(test_loc));

if ~isempty(intersect(class_train_loc,class_test_loc)) 
    fprintf('division wrong!\n');
end

end