function [BTrain,Y] = train_DSAZH_B(LTrain,A,param)

    % parameters
    nbits = param.nbits;
    maxIter = param.max_iter;
    eta = param.eta;
    gamma = param.gamma;
    
    n = size(LTrain,1);
    c = param.c; c1 = param.c1; c2 = param.c2;
    LTrain_s = LTrain(:,param.id_seenclass);
    LTrain_ex=[LTrain_s,zeros(n,c2)];
    A1 = A(param.id_seenclass,:); A2 = A(param.id_unseenclass,:); A = [A1;A2];
    C1 = NormalizeFea(A1)*NormalizeFea(A1)'; C1(C1<0.9)=0; C1(C1>=0.9)=1;
    %%%Sc = (2+e2)*LTrain_ex-1;
    %%%Si = LTrain_s*(2*C1+e1*eye(c1))*LTrain_s'-1;
    
    if ~isfield(param,'e1')
        param.e1 = floor(((n*n)-2*sum(sum(LTrain_s*C1*LTrain_s')))/(sum(sum(LTrain_s*LTrain_s'))));
        param.e2 = floor((n*c)/sum(sum(LTrain_ex))-2);
    end
    e1 = param.e1; e2 = param.e2;
    
    % initization
    randn('seed',0);
    Y = orth(randn(max(c,nbits))); Y = Y(1:c,1:nbits);
    %BTrain = sign(randn(n,nbits));
    BTrain = sign(orth(randn(n,nbits)));

    for i = 1:maxIter
        %fprintf('iteration %3d\n', i);
        
        % update SP
        SP = (A'*A+gamma/eta*eye(size(A,2))) \ (A'*Y);
        
        % update Y
        YM = nbits*((2+e2)*LTrain_ex'*BTrain - ones(c,1)*(ones(1,n)*BTrain))+eta*A*SP;
        [~,Lmd,YVV] = svd(YM'*YM);
        idx = (diag(Lmd)>1e-6);
        YV = YVV(:,idx); YV_ = orth(YVV(:,~idx));
        YU = YM*(YV/(sqrt(Lmd(idx,idx))));
        YU_ = orth(randn(max(c,nbits-length(find(idx==1))))); YU_ = YU_(1:c,1:nbits-length(find(idx==1)));
        Y = sqrt(nbits)*[YU YU_]*[YV YV_]';
        clear YM YV YV_ YU YU_ YVV
        
        % update H
        HM = LTrain_s*((2*C1+e1*eye(c1))*(LTrain_s'*BTrain))-ones(n,1)*(ones(1,n)*BTrain);
        
        Temp = HM'*HM-1/n*(HM'*ones(n,1)*(ones(1,n)*HM));
        [~,Lmd,HVV] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        HV = HVV(:,idx); HV_ = orth(HVV(:,~idx));
        HU = (HM-1/n*ones(n,1)*(ones(1,n)*HM)) *  (HV / (sqrt(Lmd(idx,idx))));
        HU_ = orth(randn(n,nbits-length(find(idx==1))));
        H = sqrt(n)*[HU HU_]*[HV HV_]';
        clear HM HU HV UU HVV
        
        % update B
        %%%BTrain = sign(sign(LTrain_s*((2*C1+e1*eye(c1))*(LTrain_s'*H))-ones(n,1)*(ones(1,n)*H))+...
                      %%%sign(((2+e2)*LTrain_ex*Y - ones(n,1)*(ones(1,c)*Y))));
        BTrain = double(((LTrain_s*((2*C1+e1*eye(c1))*(LTrain_s'*H))-ones(n,1)*(ones(1,n)*H))>0) & ((((2+e2)*LTrain_ex*Y - ones(n,1)*(ones(1,c)*Y)))>0));
        BTrain(BTrain==0) = -1;
    end

end