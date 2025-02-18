function [XTrain,LTrain,XTest,LTest,XVal,LVal,class_unseen]=split_data_ps(features,labels,dsname)
switch(dsname)
    case 'AWA1'
        c = 50; trn = 80; ten = 80;
        class_unseen = [7	9	23	24	30	31	34	41	47	50];
    case 'AWA2'
        c = 50; trn = 100; ten = 100;
        class_unseen = [7	9	23	24	30	31	34	41	47	50];
    case 'aPY'
        c = 32; trn = 50; ten = 50;
%         class_unseen = [10	13	14	15	16	17	19	20	21	23	25	29];
%         class_unseen = [10	13	14	15	16	17	19	20];
        class_unseen = [13	14	15	16	17	19	20  25];
    case 'CUB'
        c = 200; trn = 30; ten = 30;
        class_unseen = [7	19	21	29	34	36	50	56	62	68	69	72	79	80	87	88	91	95	98	100	104	108	116	120	122	124	125	129	139	141	142	150	152	157	159	160	166	167	171	174	176	179	182	185	187	189	191	192	193	195];
    case 'SUN'
        c = 717; trn = 10; ten = 10;
        class_unseen = [4	11	24	25	33	39	54	58	73	75	76	86	96	100	104	113	125	131	139	146	153	159	185	197	217	222	238	246	247	255	260	263	287	299	316	329	337	343	354	359	380	382	421	424	426	441	449	472	483	494	509	510	518	530	559	561	581	623	632	636	646	651	657	659	675	680	682	696	711	712	713	716];
    case 'ImageNet'
        c = 1000; trn = 200; ten = 200;
        class_unseen = randperm(c,200);

end

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