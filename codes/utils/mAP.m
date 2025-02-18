function map = mAP(ids, Lbase, Lquery, R)

nquery = size(ids, 2);
APx = zeros(nquery, 1);
if ~exist('R','var') || R==0
    % R = size(Lbase,1); % Configurable
    R = 5000;
end

for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    %Lbase每个被选中行与label向量相应位进行比较，相同为1，否则为0,然后按行求和，在逻辑判断
    %最终结果就是如果测试集标签向量与当前标签向量有相同标签，则为1，否则为0
    %可以用标签矩阵相乘简化
    imatch = sum(bsxfun(@eq, Lbase(idx(1:R), :), label), 2) > 0;
    %计算前R行预测正确的数量
    LX = sum(imatch);
    %对于预测前R位为真，求AP时，统计当前有几个TP
    Lx = cumsum(imatch);
    %精度precision
    Px = Lx ./ (1:R)';
    if LX ~= 0
        APx(i) = sum(Px .* imatch) / LX;
    end
end
map = mean(APx);

end
