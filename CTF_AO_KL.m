function [A,lambda,Out] = CTF_AO_KL(Y,I,F,opts)
N  = length(I);

if ~isfield(opts,'max_iter'),   opts.max_iter   = 1000;          end
if ~isfield(opts,'tol_impr');   opts.tol_impr   = 1e-7;          end

A      = opts.A0;     % Initial tensor factors
lambda = opts.l0;     % Prior of hidden variable
tol    = 1e-4;

% cost and relative cost
Out.hist_cost         = zeros(opts.max_iter,1);

iter = 1;
rows = cell(N,1);
cols = cell(N,1);

for n = 1:N
    for i=1: size(opts.marg,1)
        [~,c]= find(opts.marg{i} == n);
        if(~isempty(c))
            rows{n}= [rows{n} i];
            cols{n}= [cols{n} c];
        end
    end
end

%%%%%%%% Precompute Data %%%%%%%%
Y_p = cell(N,1);
for n = 1:N
    Y_p{n} = cell(length(rows{n}));
    for i=1:length(rows{n})
        Y_p{n}{i} = reshape(permute(Y{rows{n}(i)},[cols{n}(i) 1:cols{n}(i)-1 cols{n}(i)+1:ndims(Y{rows{n}(i)})]),I(n),[]);
    end
end

while(1)
    %%%  Solve each subproblem with Mirror Descent
    for n = 1:N
        max_iter = 1000;
        A{n}     = CTF_AO_KL_sub(Y_p{n},I,F,A,lambda,opts.marg,rows{n},cols{n},n,max_iter,tol);
    end
    
    %%% Update prior weights
    max_iter = 100;
    [lambda,out_lambda] = CTF_AO_KL_sub_lambda(Y,F,A,lambda,opts.marg,max_iter,tol);
    Out.hist_cost(iter) = out_lambda.cost;
    if iter>1
        if (iter == opts.max_iter ||  abs(Out.hist_cost(iter) - Out.hist_cost(iter-1))/abs(Out.hist_cost(iter-1)) < opts.tol_impr )
            Out.iter = iter;
            Out.hist_cost(iter+1:end) = [];
            break;
        end
        if mod(iter,200) == 0, fprintf('Iteration : %d cost : %d  \n', iter, Out.hist_cost(iter)); end
    end
    iter = iter + 1;
end
end