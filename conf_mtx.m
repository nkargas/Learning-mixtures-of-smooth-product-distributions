function [A,pur,ac] = conf_mtx(idx1,idx2)

C = length(unique(idx2));
A = zeros(C,C);
for i=1:C
    for j=1:C
        A(i,j) = sum( idx2(idx1==i)==j  );
    end
end

m = sum(A,2);
A_ = A;
for i=1:C
    A_(i,:) = A_(i,:)/m(i);
end

A_(isnan(A_)) = 0;
pur = max(A_,[],2);
pur = pur'*m/sum(m);

Cost = zeros(C,C);
for i=1:C
    for j=1:C
        Cost(i,j) = sum(A(i, [1:(j-1) (j+1):end] ));
    end
end

[rowsol,~,~,~,~] = lapjv(Cost);

idx_perm = zeros(size(idx1));
for i=1:C
    idx_perm(idx1==i) = rowsol(i);
end
ac= sum(idx_perm==idx2)/length(idx2);


