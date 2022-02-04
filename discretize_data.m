function [X_d,E] = N_discretize(X,V_size)
N    = size(X,2);
Y    = cell(N,1);
X_d  = zeros(size(X));
for n=1:N
    [Y{n},E{n}] = discretize(X(:,n),V_size);
    X_d(:,n) = Y{n};
end
end