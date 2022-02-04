function [A_new] = CTF_AO_KL_sub(Y_p,I,F,A,lambda,marg,rows,cols,n,max_iter,tol)

% Armijo rule step size!
beta  = 0.5;
sigma = 1e-3;
step  = 0.5;
%

for i=1:length(Y_p)
    nz{i} = find(Y_p{i}(:));
    Y_p{i} = Y_p{i}(nz{i});
end

Q = cell(length(rows),1);
for i=1:length(rows)
    ps   = cols(i);
    if (length(marg{rows(i)})>1)
        Q{i} = kr(A(marg{rows(i)}([end:-1:ps+1 ps-1:-1:1])))*diag(lambda);
    else
        Q{i} = lambda';
    end
end
c = [];
A_new = A{n};

cost_prev = cost(A_new,Y_p,Q,nz);
for iter = 1:max_iter
    grad_A = zeros(I(n),F);
    for i=1:length(rows)
        d           = A_new*Q{i}';
        frac        = zeros(size(d));
        frac(nz{i}) = Y_p{i}./d(nz{i});
        grad_A      = grad_A - frac*Q{i};
    end
    A_old  = A_new;
    [A_new,cost_new]  = armijo_rule(A_old,Y_p,Q,grad_A,beta,sigma,step,cost_prev,nz);
    c         = [c; cost_new];
    cost_prev = c(end);
    
    if (norm(A_old(:) - A_new(:))/norm(A_old(:)) < tol)
        break;
    end
end
end

function [A,cost_new] = armijo_rule(A_old,Y_p,Q,grad_A,beta,sigma,step,cost_prev,nz)
A_new     = A_old .* exp(-step*grad_A);
A_new     = bsxfun(@times,A_new,1./sum(A_new,1));
cost_new  = cost(A_new,Y_p,Q,nz);
d = A_old - A_new;

while sum(isnan(A_new(:)))>0 || (cost_prev-cost_new) < sigma * grad_A(:)'*d(:)
    step      = step*beta;
    A_new     = A_old .* exp(-step*grad_A);
    A_new     = bsxfun(@times,A_new,1./sum(A_new,1));
    cost_new  = cost(A_new,Y_p,Q,nz);
    d = A_old - A_new;
    if step<1e-5
        A_new = A_old;
        break
    end
end
A = A_new;
end

function c = cost(A,Y_p,Q,nz)
c = 0;
for i=1:length(Q)
    d       = A*Q{i}';
    frac    = Y_p{i}./d(nz{i});
    temp    = Y_p{i}.*log(frac(:));
    temp(isnan(temp)) = 0;
    c = c + sum(temp);
end
end