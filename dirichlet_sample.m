function [r] = dirichlet_sample(a,n)
% take a sample from a dirichlet distribution
% Returns a matrix of size length(a) times n
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
