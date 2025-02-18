%a substitute for vec2ind function on matlab of windows version
%it turn normal labels matrix L(row represent samples and column represents
%into number
%for example : [0 1 0] -> 2
%obviously, it only suits for single label datasets
function labels = vec2ind02(L)
temp = repmat(1:size(L,2),size(L,1),1);
labels = sum(L.*temp,2);
end

