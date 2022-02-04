function [y,comp] = pdf_true(l,A,S,X)

N = size(A,2);
F = length(l);

y = ones(size(X,1),1);
tmp = ones(length(y),F);

for i=1:length(y)
    for f=1:F
        for n=1:N
            tmp(i,f) = tmp(i,f) * (0.5*normpdf(X(i,n),A(f,n,1),sqrt(S(f,n,1))) + 0.5*normpdf(X(i,n),A(f,n,2),sqrt(S(f,n,2))));
        end
        tmp(i,f) = tmp(i,f)*l(f);
    end
end
[~,comp] = max(tmp,[],2);
y  = sum(tmp,2);
end
