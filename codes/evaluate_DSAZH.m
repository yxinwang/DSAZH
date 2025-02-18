function evaluation_info=evaluate_DSAZH(XKTrain,LTrain,XKTest,LTest,XKRetr,LRetr,att,param)

    if strfind(param.db_name,'AWA') & strfind(param.split,'SS')
        param.max_iter = 5;%n
        param.eta = 1;%n
        param.gamma = 0.1;%n
        param.sigma = 1;
        param.p = 1.5;
        param.e1 = 37; param.e2 = 48;
    elseif strfind(param.db_name,'AWA') & strfind(param.split,'PS')
        param.max_iter = 5;%n
        param.eta = 1;%n
        param.gamma = 0.1;%n
        param.sigma = 1;
        param.p = 1.5;
        param.e1 = 36; param.e2 = 48;
    elseif strcmp(param.db_name,'aPY') & strfind(param.split,'SS')
        param.max_iter = 5;%n
        param.eta = 1;%n
        param.gamma = 0.1;%n
        param.sigma = 10;
        param.p = 1;
        param.e1 = 20; param.e2 = 30;
    elseif strcmp(param.db_name,'aPY') & strfind(param.split,'PS')
        param.max_iter = 5;%n
        param.eta = 1;%n
        param.gamma = 0.1;%n
        param.sigma = 10;
        param.p = 1;
        param.e1 = 18; param.e2 = 30;
    elseif strcmp(param.db_name,'ImageNet')%PS
        param.e1 = 98; param.e2 = 108;
        param.max_iter = 5;%n
        param.eta = 1;%n
        param.gamma = 0.1;%n
        param.sigma = 1;
        param.p = 1.5;
        param.e1 = 98; param.e2 = 108;
    end
    
    param.c = size(LRetr,2);
    param.id_seenclass = find(sum(LTrain,1)~=0);
    param.c1 = length(param.id_seenclass);
    param.id_unseenclass = find(sum(LTrain,1)==0);
    param.c2 = length(param.id_unseenclass);
    
    % ======== Hash Learning ==========
    tic;
    
    %hash codes
    [BTrain,Y] = train_DSAZH_B(LTrain,att',param);
    
    %hash functions
    XP = train_DSAZH_func(XKTrain,LTrain,BTrain,att',param);

    %domain-aware threshold
    Bc = sign(LTrain(:,param.id_seenclass)'*BTrain -eps);
    param.thre = floor(param.p*mean(sum(abs(sign(LTrain(:,param.id_seenclass)*Bc)-sign(XKTrain*XP)),2)/2));
    
    evaluation_info.trainT=toc;
    

    % ============= Evaluate ================
    tic;
    BRetr = compactbit((XKRetr*XP)>0);
    BTest = compactbit((XKTest*XP)>0);

    DHamm = hammingDist(BTest, BRetr);
    preseen_idx = min(hammingDist(compactbit(Bc>0), BRetr)) <= param.thre;
    DHamm(:,preseen_idx) = param.nbits+1;
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.MAP = mAP(orderH', LRetr, LTest, param.top_R);
    fprintf('MAP: %f\n',evaluation_info.MAP);
    [evaluation_info.precision,evaluation_info.recall] = precision_recall(orderH', LRetr, LTest);
    evaluation_info.Precision = precision_at_k(orderH',LRetr, LTest,param.top_K);
    
    evaluation_info.testT=toc;
    
    
    evaluation_info.param = param;
    
end