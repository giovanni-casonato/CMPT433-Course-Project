
%Error back-propagation, from output neuron to hidden layer

function [hid_errmat1, hid_errmat2] = ErrBackProp(hid_outmat, hidneur_w2, outneur_w, z_d, hidneur_num, num_vars, abs_hid_outmat1, abs_hid_outmat2)

%hid_outmat = (N x n) matrix of hidden neuron outputs, where
%N = number of learning samples, n = number of hidden neurons

%outneur_w = (n+1 x 1) vector of output neuron weights

%z_d = (N x 1) vector of desired network outputs, expressed as weighted
%sums lying on the unit circle

N = size(hid_outmat, 1);
%n = size(hid_outmat, 2);

hidneur_num1 = hidneur_num(1);
hidneur_num2 = hidneur_num(2);

%append a column of 1s to hid_outmat from the left, making it a (N x n+1) matrix
col_app(1:N) = 1;
col_app = col_app.';
hid_outmat2 = [col_app hid_outmat];


%Compute the weighted sums of the output neuron for all N samples
z_c = hid_outmat2 * outneur_w;
%z_ctmp = z_c;
%Move output to the unit circle
%z_c = z_c ./ abs(z_c);

%for ii=1:N
%    abs_zc = abs(z_c(ii));
%    if (abs_zc ~= 0)
%        z_c(ii) = z_c(ii) / abs_zc;
        
%    end
%end

%if (sum(isnan(z_c)) > 0)
%    flag = 1;
%end

%Compute network error for all samples
net_err = z_d - z_c;
% net_err = z_d - z_ctmp;


%Arguments of the outputs
%current_phase = mod(angle(z_c), 2*pi);


%current_labels = floor(current_phase ./ sec_size);

%for ii=1:N
    %if (current_labels(ii) == y_d(ii))
    %     net_err(ii) = 0;
    %end
    
    %ang_err = abs(current_phase(ii) - phase_d(ii));
    
    %if (ang_err > pi)
        
    %    ang_err = 2*pi - ang_err;
    %end
    
%end


%Compute the error of the output neuron
outneur_err = net_err ./ (hidneur_num2+1);

%outneur_err = net_err;


%Construct a (N x n) matrix whose columns are equal to outneur_err
%outneur_errmat = zeros(N, n);
%    for jj=1:n
        
%        outneur_errmat(:, jj) = outneur_err;
%    end

%Construct a diagonal (n x n) matrix containing the reciprocal weights w1 to wn of the output neuron
%outneur_rWmat = diag(1 ./ outneur_w(2:n+1));

%Construct a (1 x n) vector containing the reciprocal weights w1 to wn of
%the output neuron
%outneur_rWvec(1, 1:n) = 1 ./ outneur_w(2:n+1);
outneur_rWvec(1, 1:hidneur_num2) = (outneur_w(2:hidneur_num2+1)).^-1;

% errors of the 2nd hiddel layer neurons are equal to the output neuron's
% errors times corresponding reciprocal weights of the ourpur neuron
hid_errmat2 = outneur_err * outneur_rWvec;

  % normalization of the errors of the 2nd hidden layer neurons by the
  % number of weights of each second hidden layer neuron - sharing the 
  % error among inputes
  %hid_errmat2 = hid_errmat2 ./ (hidneur_num1 + 1);

    hid_errmat2 = hid_errmat2 ./ (hidneur_num2 + 1);
  
 

%Construct a (1 x n) vector containing the reciprocal weights w1 to wn of
%the 2nd hiddel layer neurons
hidneur2_rWvec = (hidneur_w2(2:hidneur_num1 + 1, :).^-1);
% errors of the 1st hiddel layer neurons are equal to the weighted sums of
% the 2nd hidden layer neurons with the corresponding reciprocal weights
hid_errmat1 = hid_errmat2 * hidneur2_rWvec.';

% Normalization of the 1st hidden layer neurons errors by the # of neurons
% in the 2nd hidden layer
 %hid_errmat1 = hid_errmat1 ./ hidneur_num2;

 hid_errmat1 = hid_errmat1 ./ (hidneur_num1 + 1);

 % normalization of the 1st hidden layer neurons errors by the number of
 % neurons in the 2nd hidden layer where from these errors were
 % backpropagated
 % !!!!!!!
 % hid_errmat1 = hid_errmat1 ./ hidneur_num2;
 
 
  % normalization of the errors of the 2nd hidden layer neurons by the
  % number of weights of each second hidden layer neuron - sharing the 
  % error among inputes
%  hid_errmat2 = hid_errmat2 ./ (hidneur_num1 + 1);
 

  % normalization of the errors of the 1st hidden layer neurons by the
  % number of weights of each 1st hidden layer neuron - sharing the 
  % error among inputes
%  hid_errmat1 = hid_errmat1 ./ (num_vars + 1);

  % normalization of the errors by abs values of the current weighted sums
  hid_errmat1 = hid_errmat1./abs_hid_outmat1;
  hid_errmat2 = hid_errmat2./abs_hid_outmat2;

end


    
%Compute the error of each hidden neuron for all samples. The result is a
%(N x n) matrix, where each column represent a single hidden neuron
%hid_errmat = outneur_errmat * outneur_rWmat;

%---
