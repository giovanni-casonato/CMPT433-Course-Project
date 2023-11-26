
%Weights adjustment for output neuron

function [w_adj, z_c] = OutNeuron_weightadj(X, z_d, w)

%X = matrix of output neuron inputs (N x n), where N=number of learning
%samples, n = number of hidden neurons

%w = (n+1 x 1) vector of weights of the output neuron
%phase_d = (N x 1) vector of desired network outputs for all N learning
%samples, expressed as phase

%------
%n = size(X, 2);
N = size(X, 1);

%append a column of 1s to X from the left, making it a (N x n+1) matrix
col_app(1:N) = 1;
col_app = col_app.';
X = [col_app X];

%Compute the weighted sums of the output neuron for all N samples
z_c = X * w;

%Move output to the unit circle
% z_c = z_c ./ abs(z_c);


%Compute network error for all samples
net_err = z_d - z_c;



%Use LLS to compute a weights adjustment vector adj_val
%adj_vec = X \ delta;

%if (sum(sum(isnan(X))) > 0)
%    flag = 1;
%end

%Compute the full SVD of X
%[U,S,V] = svd(X);

%M = n+1
%M = length(w);

%Retain only the first M columns of U, and first M rows of S
%U_hat = U(:, 1:M);
%S_hat = S(1:M, :);

%Construct the pseudo-inverse of S
%S_hpinv = diag(1 ./ diag(S_hat));

%Construct the pseudo-inverse of X
%X_pinv = V * S_hpinv * U_hat';

X_pinv = pinv(X);

%LLS: apply X_pinv to delta
%adj_vec = X_pinv * net_err;

M = length(w);
% adj_vec = X_pinv * ( net_err ./ abs(z_c) ./ M);
 adj_vec = X_pinv * ( net_err ./ M);  

%the new weights are given by 
w_adj = w + adj_vec;

%Construct a vector of current weighted sums for N samples
z_c = X * w_adj;





