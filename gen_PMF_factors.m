function [A,prior] = gen_PMF_factors(I,F)
% Generate factors
N     = length(I);
A     = cell(N,1);
scale = cell(N,1);
for n = 1:N
    A{n}     = rand(I(n),F);
    scale{n} = sum(A{n},1);
    A{n}     = bsxfun(@times,A{n},1./sum(A{n},1));
end
prior = rand(F,1); prior = prior/sum(prior);
end