clc
clearvars
close all

addpath('./tensorlab')

F_true   = 10;
F_inner  = 2;

V_size  = 15;                % Number of bins
N       = 10;                % Number of variables
I       = V_size*ones(1,N);  % Size of probability tensor
N_sim   = 5;                 % Number of simulations
I_itr   = 3;                 % Number of initilizations

n_samples    = [1000 2000 3000 5000 7000 10000];
l_samples    = length(n_samples);

% EM estimated parameters
A_est_GMM  = cell(N_sim,l_samples,I_itr); S_est_GMM  = cell(N_sim,l_samples,I_itr);
l_est_GMM  = cell(N_sim,l_samples,I_itr); cost_GMM   = zeros(N_sim,l_samples,I_itr);

% True parameters
A_true = cell(N_sim,1); S_true = cell(N_sim,1); l_true = cell(N_sim,1);

% KL estimated parameters
A_est_kl = cell(N_sim,l_samples,I_itr); l_est_kl = cell(N_sim,l_samples,I_itr);
Out_kl   = cell(N_sim,l_samples,I_itr); cost_kl  = zeros(N_sim,l_samples,I_itr);

% FR estimated parameters
A_est_fr = cell(N_sim,l_samples,I_itr); l_est_fr = cell(N_sim,l_samples,I_itr);
Out_fr   = cell(N_sim,l_samples,I_itr); cost_fr  = zeros(N_sim,l_samples,I_itr);

% Edges for discretization
E = cell(N_sim,l_samples);
for s=1:N_sim
    for n_it=1:l_samples
        E{s,n_it} = cell(N,1);
    end
end

%%%%%%%%%%%%%%%%%%%%% Algorithm Options %%%%%%%%%%%%%%%%%%%%%%
marg = combnk(1:N,3);
marg = marg(1:5:end,:);  % Keep only a fraction
marg = num2cell(marg,2);

for s=1:N_sim
    fprintf('=================  Simulation : %d ================== \n',s);
    %%%%%%%%%%% Prior probability
    alpha  = 10;
    l_true{s} = dirichlet_sample(alpha*ones(1,F_true),1)';
    dist  = 2;                                 % Distance between Gaussians
    
    for n=1:N
        for f1=1:F_true
            pos = -dist+(2*dist)*rand();
            A_true{s}(f1,n,1) = pos+5*rand();
            A_true{s}(f1,n,2) = pos-5*rand();  % Mixture Component Means
        end
    end
    S_true{s} = 3*rand(F_true, N, F_inner) + 1;
    
    for n_it=1:l_samples
        % Create data and discretize
        [X,c] = pdf_sample(l_true{s},A_true{s},S_true{s},n_samples(n_it));
        [X_d,E{s,n_it}] = discretize_data(X,V_size);
        
        I = ones(1,N)*V_size;
        Y = get_obs_marg(X_d,marg,I);
        
        % Run algorithms
        for inner = 1:I_itr
            fprintf('Inner iteration : %d \n',inner);
            %%% Options
            opts = {}; opts.marg = marg; opts.max_iter = 500; opts.tol_impr = 1e-5;
            for n = 1:N, opts.constraint{n} = 'simplex_col'; end; opts.constraint{N+1} = 'simplex';
            
            [opts.A0,~] = gen_PMF_factors(I,F_true);
            opts.l0 = ones(F_true,1)*1/F_true;
            
            %Algorithm 1
            [A_est_kl{s,n_it,inner},l_est_kl{s,n_it,inner},Out_kl] = CTF_AO_KL(Y,I,F_true,opts);
            [A_est_fr{s,n_it,inner},l_est_fr{s,n_it,inner},Out_fr] = CTF_AO_FRO(Y,I,F_true,opts);
            
            cost_kl(s,n_it,inner) = Out_kl.hist_cost(end);
            cost_fr(s,n_it,inner) = Out_fr.hist_cost(end);
            
            except_ = true;
            while(except_)
                try
                    GMM_dist = fitgmdist(X,F_true,'CovarianceType','diagonal','Options',statset('MaxIter',5000),'Start','randSample');
                    A_est_GMM{s,n_it,inner}  = GMM_dist.mu;
                    S_est_GMM{s,n_it,inner}  = GMM_dist.Sigma;
                    l_est_GMM{s,n_it,inner}  = GMM_dist.ComponentProportion;
                    cost_GMM(s,n_it,inner)   = GMM_dist.NegativeLogLikelihood;
                    except_ = false;
                catch exception
                end
            end
        end
    end
end

% close all
% Stochastic integration
er_kl  = zeros(N_sim,length(n_samples));
er_fr  = zeros(N_sim,length(n_samples));
er_em  = zeros(N_sim,length(n_samples));

ac_kl  = zeros(N_sim,length(n_samples));
ac_fr  = zeros(N_sim,length(n_samples));
ac_em  = zeros(N_sim,length(n_samples));
ac_oracle = zeros(N_sim,length(n_samples));

N_points = 400;
%%% Gather all information
for s=1:N_sim
    % Generate M points
    M = 1000;
    
    [X_s,X_clusters]    = pdf_sample(l_true{s},A_true{s},S_true{s},M);
    [y_true,idx_oracle] = pdf_true(l_true{s},A_true{s},S_true{s},X_s);
    
    for n_it=1:l_samples
        % Find best iteration for each algorithm
        [~,ind_kl] = min(squeeze(cost_kl(s,n_it,:)));
        [~,ind_fr] = min(squeeze(cost_fr(s,n_it,:)));
        [~,ind_em] = min(squeeze(cost_GMM(s,n_it,:)));
        
        % Compute accuracy
        gm_em = gmdistribution(A_est_GMM{s,n_it,ind_em}, S_est_GMM{s,n_it,ind_em},l_est_GMM{s,n_it,ind_em});
        idx_em = cluster(gm_em,X_s);
        [~,~,ac_em(s,n_it)] = conf_mtx(idx_em,X_clusters);
        
        idx_kl      = cluster_data(A_est_kl{s,n_it,ind_kl},l_est_kl{s,n_it,ind_kl},X_s,E{s,n_it},N_points,V_size);
        [~,~,ac_kl(s,n_it)] = conf_mtx(idx_kl,X_clusters);
        
        idx_fr = cluster_data(A_est_fr{s,n_it,ind_fr},l_est_fr{s,n_it,ind_fr},X_s,E{s,n_it},N_points,V_size);
        [~,~,ac_fr(s,n_it)] = conf_mtx(idx_fr,X_clusters);
                
        y_kl    = pdf_estimate(A_est_kl{s,n_it,ind_kl},l_est_kl{s,n_it,ind_kl},X_s,E{s,n_it},N_points,V_size);
        y_fr    = pdf_estimate(A_est_fr{s,n_it,ind_fr},l_est_fr{s,n_it,ind_fr},X_s,E{s,n_it},N_points,V_size);
        y_em    = pdf(gm_em,X_s);
        
        % KL-divergence (MC integration)
        er_kl(s,n_it)  = sum(1/M * log(y_true./y_kl));
        er_fr(s,n_it)  = sum(1/M * log(y_true./y_fr));
        er_em(s,n_it)  = sum(1/M * log(y_true./y_em));
        
    end
end

figure
hold on
plot(n_samples, mean(er_kl,1),'s--','linewidth',2,'markersize',16)
plot(n_samples, mean(er_fr,1),'o--','linewidth',2,'markersize',16)
plot(n_samples, mean(er_em,1),'d--','linewidth',2,'markersize',16)
xlabel('Number of samples','interpreter','latex')
ylabel('KL divergence','interpreter','latex')
legend( {'Alg-KL','Alg-Frob','EM GMM'},'interpreter','latex')
title('Number of components $F=10$','interpreter','latex')
hold off

figure
hold on
plot(n_samples, mean(ac_kl,1),'s--','linewidth',2,'markersize',16)
plot(n_samples, mean(ac_fr,1),'o--','linewidth',2,'markersize',16)
plot(n_samples, mean(ac_em,1),'d--','linewidth',2,'markersize',16)
xlabel('Number of samples','interpreter','latex')
ylabel('Clustering accuracy','interpreter','latex')
legend( {'Alg-KL','Alg-Frob','EM GMM'},'interpreter','latex','Location','southeast')
title('Number of components $F=10$','interpreter','latex')
hold off


