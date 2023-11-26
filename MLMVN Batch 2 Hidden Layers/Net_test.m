
function [ang_RMSE, current_phase, y_d] = Net_test(Input, hidneur_weights1, hidneur_weights2, outneur_w)

% This function utilizes the MLMVN test after batch learning algorithm for
% a network containing 2 hidden layers and a single output neuron
% This version of the function supports only continuous inputs and
% continuous output

% Description of input parameters:
% Input - matrix of MVN inputs (N x (n+1)), where N=number of learning
% samples, n = number of inputs, the last column of this matrix represent 
% desired outputs. Each row of this matrix is a learning sample - inputs
% followed by the desired output
%
% hidneur_weights1 - weights of all neurons from the 1st hidden layer
%
% hidneur_weights2 - weights of all neurons from the 2nd hidden layer
%
% outneur_w - weights of output neuron

%Description of output parameters:
%
% ang_RMSE - test angular RMSE
% current_phase - actual outputs
% y_d - desired outputs taken from the last column of Input
%


%Determine the number of testing samples and number of inputs
[N, m] = size(Input);
n = m-1;
X = Input(:,1:n);
y_d = Input(:,m);

%Convert input values into complex numbers on the unit circle
X = exp(1i .* X);


%Determine sector size (angle)
%sec_size = 2*pi / num_classes;


%append a column of 1s to X from the left
%app_X
col_app(1:N) = 1;
col_app = col_app.';
app_X = [col_app X];

%Compute the output of hidden neurons for all samples
hid_outmat1 = app_X * hidneur_weights1;

%Move outputs to the unit circle
hid_outmat1 = hid_outmat1 ./ abs(hid_outmat1);

    app_X1 = [col_app hid_outmat1];
    
    %Compute the output of the 2nd hidden neurons for all samples
    hid_outmat2 = app_X1 * hidneur_weights2;
    
    %Move outputs to the unit circle
    hid_outmat2 = hid_outmat2 ./ abs(hid_outmat2);

%append a column of 1s to hid_outmat
hid_outmat2 = [col_app hid_outmat2];

%Compute the output of the network
z_outneur = hid_outmat2 * outneur_w;


current_phase = mod(angle(z_outneur), 2*pi);

    ang_RMSE = 0;
    
    for ii=1:N
        
        ang_err = abs(y_d(ii) - current_phase(ii));
        
        if (ang_err > pi)
        
            ang_err = 2*pi - ang_err;
        end
        
        ang_RMSE = ang_RMSE + ang_err^2;
    end

    ang_RMSE = sqrt(ang_RMSE / N);
    
    plot(y_d, 'or');
    hold on
    plot(current_phase, '*b');
    hold off
    
    
end


%current_labels = floor(current_labels ./ sec_size);

%sovpad = 0;

%for ii=1:N
%    if (current_labels(ii) == y_d(ii))
%        sovpad = sovpad + 1;
%    end
%end

%classif_rate = sovpad / N;

%FOR GLASS ONLY
%sec_popal = current_labels(1);

%disp(['sec desired ', num2str(y_d)]);
%disp(['sec popal ', num2str(sec_popal)]);
%disp(' ');




