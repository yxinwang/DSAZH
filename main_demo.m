close all; clear; clc;
addpath(genpath('./codes/'));

db = {'AWA2'}; splits = {'SS','PS'};
loopnbits = [16,32,64,128,256];

param.top_R = 5000;
param.top_K = 2000;
param.pr_ind = [1:50:1000,1001];
param.pn_pos = [1:100:2000,2000];

for spi = 1:2
for dbi = 1

    db_name = db{dbi}; param.db_name = db_name;
    param.split = splits{spi};
    
    diary(['./results/commandWindow_',db_name,'.txt']);
    diary on;
    
	
    %% load dataset
    param.seed = 0; rng('default'); rng(param.seed);

	load(['./datasets/',db_name,'.mat']);
	
	if strfind(param.split,'SS')
		result_name = [result_URL 'final_' db_name '_SS_result' '.mat'];
		[XTrain,LTrain,XTest,LTest,XRetr,LRetr,class_unseen]=split_data(features,labels,db_name);
	else
		result_name = [result_URL 'final_' db_name '_PS_result' '.mat'];
		[XTrain,LTrain,XTest,LTest,XRetr,LRetr,class_unseen]=split_data_ps(features,labels,db_name);
	end
        
    lTrain = vec2ind02(LTrain); lRetr = vec2ind02(LRetr); lTest = vec2ind02(LTest);
    clear features labels
    
    
    %% Kernel representation
    param.nXanchors = 1000;
    anchor_idx = randsample(size(XTrain,1), param.nXanchors);
    XAnchors = XTrain(anchor_idx,:);
    [XKTrain,bandwidth,mvec] = RBF_kernel(XTrain,XAnchors);
    [XKTest,~,~] = RBF_kernel(XTest,XAnchors,bandwidth,mvec);
    [XKRetr,~,~] = RBF_kernel(XRetr,XAnchors,bandwidth,mvec);
    clear bandwidth mvec;
	
    
    %% Methods
    eva_info = cell(1,length(loopnbits));
    
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii));
        param.nbits = loopnbits(ii);
        
		fprintf('\n......%s start...... \n', 'DSAZH');
		DSAZHparam = param;
		eva_info_ = evaluate_DSAZH(XKTrain,LTrain,XKTest,LTest,XKRetr,LRetr,att,DSAZHparam);
		eva_info{1,ii} = eva_info_;
		clear eva_info_
    end
    
    %% Results
    for ii = 1:length(loopnbits)
		MAP{1,ii} = eva_info{1,ii}.MAP;
		trainT{1,ii} = eva_info{1,ii}.trainT; 
    end
    
    result_URL = './results/';
    if ~isdir(result_URL)
        mkdir(result_URL);
    end
    save(result_name,'eva_info','param','loopnbits','MAP','trainT','XTrain','XTest','XRetr','LTrain','LTest','LRetr');
    
    diary off;
end
end