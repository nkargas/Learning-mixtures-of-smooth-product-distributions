function V = computeV_lambda(marg,A,Y,F)
V = zeros(F,1);
for i = 1:size(marg,1)
    V = V + kr(A(marg{i}(end:-1:1)))'*Y{i}(:);
end