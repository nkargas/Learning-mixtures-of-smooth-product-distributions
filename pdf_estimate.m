function y = pdf_estimate(A,l,X,E,N_points,V_size)

N = size(A,1);
F = length(l);

tmp = ones(size(X,1),F);
for f=1:F
    for n=1:N
        tmp(:,f) = tmp(:,f) .* get_conditional(X(:,n)',E{n},A{n}(:,f),N_points,V_size) ;
    end
    tmp(:,f) = tmp(:,f)*l(f);
end
y  = sum(tmp,2);
end



