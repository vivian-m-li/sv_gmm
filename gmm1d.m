%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% general implementation of a 3-mode gaussian mixture model
% created 23 May 2019
% by Aaron Clauset ( aaronclauset@gmail.com )

clear all; close all; clc;

% 1. Generate synthetic data from a 3-mode guassian mixture model

% parameter values and sample size
n  = 10000;         % sample size
mu = [0 5 10];      % mode means
vr = [1 1 1];       % mode variances
p  = [0.2 0.5 0.3]; % mixing parameters

% assign labels
nk = floor(p.*n);
z          = [1.*ones(nk(1),1); 2.*ones(nk(2),1); 3.*ones(nk(3),1)];
x1 = mu(1) + randn(nk(1),1).*sqrt(vr(1));
x2 = mu(2) + randn(nk(2),1).*sqrt(vr(2));
x3 = mu(3) + randn(nk(3),1).*sqrt(vr(3));
x  = [x1; x2; x3];

% visualize the data and the generative model
ux = (min(x):0.25:max(x));
hx = hist(x,ux);
figure(1);
plot(ux,hx,'bo'); hold on;
for i=1:numel(mu)
    plot(ux,(n.*p(i)/4).*pdf('norm',ux,mu(i),sqrt(vr(i))),'r-');
end
hold off;

% [1] logL = -581649.30
%      mu = [0.00 5.00 10.00]
%      vr = [1.00 1.00 1.00]
%      p  = [0.20 0.50 0.30]

z_true  = z;
mu_true = mu;
vr_true = vr;
p_true  = p;

% 2. Estimate the parameters via EM

% initial conditions
mu    = [min(x) median(x) max(x)];      % initial means
vr    = [1 1 1];       % initial variances
p     = ones(1,3)./3;  % initial p_k proportions
logL  = zeros(10,1);   % logL values

% initial log-likelihood
%   logL  = sum_{i=1}^n log( sum_{k=1}^K  p_k * N(xi|uk,sk)  )
Nxi     = [pdf('norm',x,mu(1),sqrt(vr(1))) pdf('norm',x,mu(2),sqrt(vr(2))) pdf('norm',x,mu(3),sqrt(vr(3)))];
logL(1) = sum( log( p * Nxi ));

fprintf('[%i] logL = %8.2f\n',1,logL(1));
fprintf('     mu = [%4.2f %4.2f %4.2f]\n',mu);
fprintf('     vr = [%4.2f %4.2f %4.2f]\n',vr);
fprintf('     p  = [%4.2f %4.2f %4.2f]\n',p);

ux = (min(x):0.25:max(x));
hx = hist(x,ux);
figure(1);
plot(ux,hx,'bo'); hold on;
for i=1:numel(mu)
    plot(ux,(n.*p(i)/4).*pdf('norm',ux,mu(i),sqrt(vr(i))),'r-');
end
hold off;
drawnow

% run the EM algorithm
gz = zeros(n,numel(mu)); % initial posteriors
jj = 2;                  % initialize loop
while jj<=numel(logL)
    % E-step : calculate the posterior probabilities
    % gamma_zi(k) = Pr(Zi = k | xi) = pi(k)*N(xi,mu_k,si_k) / sum_k pi(k)*N(xi,mu_k,si_k)
    for i=1:numel(x)
        gz(i,:) = p.*pdf('norm',x(i),mu,sqrt(vr));
        gz(i,:) = gz(i,:)./sum(gz(i,:));
    end
    
    % M-step : estimate gaussian parameters
    % n_k  = sum_{i=1}^n gamma_zi(k)
    nk = sum(gz,1);
    % mu_k = (1/n_k) * sum_{i=1}^n gamma_zi(k)*xi
    for j=1:numel(mu)
        mu(j) = (1./nk(j)) .* sum(gz(:,j).*x);
    end
    % sg_k = (1/n_k) * sum_{i=1}^n (gamma_zi(k) * (xi-mu_k)^2
    for j=1:numel(vr)
        vr(j) = (1./nk(j)) .* sum( gz(:,j).*(x-mu(j)).^2 );
    end
    % p_k = (n_k/n)
    p = nk./n;
    
    % update likelihood
    Nxi     = [pdf('norm',x,mu(1),sqrt(vr(1))) pdf('norm',x,mu(2),sqrt(vr(2))) pdf('norm',x,mu(3),sqrt(vr(3)))];
    logL(jj) = sum( log( p * Nxi ));

    fprintf('[%i] logL = %8.2f\n',jj,logL(jj));
    fprintf('     mu = [%4.2f %4.2f %4.2f]\n',mu);
    fprintf('     vr = [%4.2f %4.2f %4.2f]\n',vr);
    fprintf('     p  = [%4.2f %4.2f %4.2f]\n',p);

    % visualize the current model
    figure(1);
    plot(ux,hx,'bo'); hold on;
    for i=1:numel(mu)
        plot(ux,(n.*p(i)/4).*pdf('norm',ux,mu(i),sqrt(vr(i))),'r-');
    end
    hold off;
    drawnow

    % convergence check
    if (logL(jj)-logL(jj-1))<0.05, break; end
    jj = jj+1;
    
end

% plot the likelihood function over time
figure(2);
plot((1:numel(logL)),logL,'bo-');
xlabel('iterations')
ylabel('log-likelihood')

% visualize the final model
ux = (min(x):0.25:max(x));
hx = hist(x,ux);
figure(1);
plot(ux,hx,'bo'); hold on;
for i=1:numel(mu)
    plot(ux,(n.*p(i)/4).*pdf('norm',ux,mu(i),sqrt(vr(i))),'r-');
end
hold off;
xlabel('x value')
ylabel('density')
