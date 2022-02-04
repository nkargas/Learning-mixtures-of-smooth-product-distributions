function [Y,sum_y] = get_obs_marg(Dataset,marg,I)
Y  = cell(size(marg,1),1);
sum_y = zeros(size(marg,1),1);

for i=1:size(marg,1)
    if (length(marg{i})>1)
        Y{i} = zeros(I(marg{i}));
        tmp = Dataset(all(Dataset(:,marg{i}) ~=0 ,2) , marg{i});
        [u,~,ic]=unique(tmp,'rows');
        C =[u histc(ic,1:size(u,1))];
        for c = 1:size(C,1)
            subscript = num2cell(C(c,1:end-1));
            ind = sub2ind(I(marg{i}),subscript{:});
            Y{i}(ind) = C(c,end);
        end
        sum_y(i) = sum(Y{i}(:));
        Y{i} = Y{i} / sum(Y{i}(:));
    else
        Y{i} = zeros(I(marg{i}),1);
        tmp = Dataset(all(Dataset(:,marg{i}) ~=0 ,2) , marg{i});
        [u,~,ic]=unique(tmp,'rows');
        C =[u histc(ic,1:size(u,1))];
        for c = 1:size(C,1)
            Y{i}(C(c,1)) = C(c,end);
        end
        sum_y(i) = sum(Y{i}(:));
        Y{i} = Y{i} / sum(Y{i}(:));
    end
end
end