function [X,comp] = pdf_sample(l,A_true,S_true,n_samples)

[F, N, F_inner] = size(A_true);
X     = zeros(n_samples,N);
comp  = sum(repmat(rand(1,n_samples),F,1) > repmat(cumsum(l),1,n_samples),1)'+1;

for n=1:N
    for i=1:F
        ind = find(comp==i);
        for ii=1:length(ind)
            f_inner= randi(1:F_inner);
            X(ind(ii),n) = A_true(i,n,f_inner) + randn*sqrt(S_true(i,n,f_inner));
        end
    end
end
end
