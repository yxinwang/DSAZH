function XP = train_DSAZH_func(XTrain,LTrain,BTrain,A,param)

    %parameters
    sigma = param.sigma;

    LTrain_s = LTrain(:,param.id_seenclass);
    
    A1 = A(param.id_seenclass,:);
    A2 = A(param.id_unseenclass,:);
    C2 = NormalizeFea(A2)*NormalizeFea(A1)';
    
    [~,param.id_boostclass] = max(C2,[],2);
    param.id_boostclass = unique(param.id_boostclass);
    
    param.is_boostsample = [];
    for cii = 1:length(param.id_boostclass)
        tttmp = find(LTrain_s(:,param.id_boostclass(cii)) == 1);
        ttttmp = tttmp(randperm(length(tttmp),0.5*length(tttmp)));
        param.is_boostsample = [param.is_boostsample;ttttmp];
    end

    XTrain_b = XTrain(param.is_boostsample,:);
    BTrain_b = BTrain(param.is_boostsample,:);

    XP = (XTrain'*XTrain + XTrain_b'*XTrain_b + sigma*eye(size(XTrain,2))) ...
        \ (XTrain'*BTrain + XTrain_b'*BTrain_b);

    %XP = (XTrain'*XTrain+sigma*eye(size(XTrain,2))) \ (XTrain'*BTrain);

end