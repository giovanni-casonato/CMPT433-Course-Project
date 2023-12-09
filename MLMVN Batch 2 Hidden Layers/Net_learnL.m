function [hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learnL(Input, Valid, hidneur_num, GlobalThreshold)
% This function utilizes the MLMVN batch learning algorithm for a network
% containing 2 hidden layers and a single output neuron
% Learning requires a validation set and stops when the validation error is
% minimized
% This version of the function supports only continuous inputs and
% continuous output

% Description of input parameters:
% Input - matrix of MVN inputs (N x (n+1)), where N=number of learning
% samples, n = number of inputs, the last column of this matrix represent 
% desired outputs. Each row of this matrix is a learning sample - inputs
% followed by the desired output
%
% Valid - a validation set - matrix of MVN inputs (N x (n+1)), where
% N=number of validation samples, n = number of inputs, the last column of
% this matrix represent desired outputs. Each row of this matrix is a
% vlidation sample - inputs followed by the desired output
%
% hidneur_num - a vector containing 2 elements: hidneur_num(1)= # of
% neurons in the 1st hidden layer, hidneur_num(2) = # of neurons in the 2nd
% hidden layer
%
% GlobalThreshold - a threshold for ang_RMSE. The learning process stops as
% soon as ang_RMSE <= GlobalThreshold holds

%Description of output parameters:
% hidneur_weights1 - weights of all neurons from the 1st hidden layer
% hidneur_weights2 - weights of all neurons from the 2nd hidden layer
% outneur_w - weights of output neuron
% iterations - the resulting number of learning iterations


%X = matrix of MVN inputs (N x n), where N=number of learning
%samples, n = number of input variables

[N, m] = size(Input);
n = m-1; % n = # of inputs
X = Input(:,1:n); % X is a matrix of inputs
y_d = Input(:,m); % y_d is a vector-column of desired outputs

% Extraction of the # of neurons on the 1st and 2nd hidden layers
hidneur_num1 = hidneur_num(1);
hidneur_num2 = hidneur_num(2);

%y_d = (N x 1) vector of desired network outputs, expressed as class labels

%hidneur_num = number of hidden neurons


%Use the clock to set the stream of random numbers
%RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));


%Convert input values into complex numbers on the unit circle
X = exp(1i .* X);

%Convert output values into complex numbers on the unit circle
znet_d = exp(1i .* y_d);

%Determine the number of learning samples
%N = size(X, 1);

%Determine the number of input variables n
%n = size(X, 2);

%Generate random weights for the 1st hidden layer neurons:
hidneur_weights1 = zeros(n+1, hidneur_num1);

for hh = 1 : hidneur_num1

    %real part (interval -0.5 to 0.5)
    w_re = rand(n+1, 1) - 0.05;
    %imaginary part (interval -0.5 to 0.5)
    w_im = rand(n+1, 1) - 0.05;

    %Construct a weights vector, dimensions (n+1 x 1)
    hidneur_weights1(:, hh) = w_re + 1i .* w_im;

end

%Generate random weights for the 2nd hidden layer neurons:
hidneur_weights2 = zeros(hidneur_num1+1, hidneur_num2);

for hh = 1 : hidneur_num2

    %real part (interval -0.5 to 0.5)
    w_re = rand(hidneur_num1+1, 1) - 0.05;
    %imaginary part (interval -0.5 to 0.5)
    w_im = rand(hidneur_num1+1, 1) - 0.05;

    %Construct a weighting vector, dimensions (n+1 x 1)
    hidneur_weights2(:, hh) = w_re + 1i .* w_im;

end


%Generate random weights for the output neuron:
%real part (interval -0.5 to 0.5)
w_re = rand(hidneur_num2+1, 1) - 0.5;
%imaginary part (interval -0.5 to 0.5)
w_im = rand(hidneur_num2+1, 1) - 0.5;
outneur_w = w_re + 1i .* w_im;

%------------------------------------------------------------------------



%Convert desired network output values (y_d), given as class labels, into
%desired phase values (phase_d):
%phase_d(1 : N, 1) = y_d .* (2*pi/num_classes);
%
%for ii=1:N
%        if (phase_d(ii) < 0)
%            phase_d(ii) = phase_d(ii) + 2*pi;
%        end
%end

%Determine sector size (angle)
%sec_size = 2*pi / num_classes;

%Shift all desired phase values by half-sector clockwise
%phase_d = phase_d + sec_size/2;


%Construct a vector of desired network outputs (lying on the unit circle)
%znet_d = exp(1j .* phase_d);


%append a column of 1s to X from the left, yielding a (N x n+1) matrix
%app_X
col_app(1:N) = 1;
col_app = col_app.';
app_X = [col_app X];

%Determine the number of testing samples and number of inputs
[N1, m1] = size(Valid);
n1 = m1-1;
XV = Valid(:,1:n1);
y_dV = Valid(:,m1);

%Convert input values into complex numbers on the unit circle
XV = exp(1i .* XV);

%append a column of 1s to X1 from the left
%app_X1
col_app1(1:N1) = 1;
col_app1 = col_app1.';
app_XV = [col_app1 XV];


%Pre-compute the SVD of app_X and the pseudo-inverse of app_X. The latter
%will be used during LLS adjustment of hidden neuron weights.
%Compute the full SVD of X
%[U,S,V] = svd(app_X);

%Let M = n+1
%M = n+1;

%Retain only the first M columns of U, and first M rows of S
%U_hat = U(:, 1:M);
%S_hat = S(1:M, :); %S_hat becomes an M x M square matrix

%Construct the pseudo-inverse of S
%S_hpinv = diag(1 ./ diag(S_hat));

%Construct the pseudo-inverse of X
%X_pinv = V * S_hpinv * U_hat';

%Construct the pseudo-inverse matrix of X
X_pinv = pinv(app_X);

% Counter of iterations
iterations = 0;
% Counter of the learning samples with the errors
nesovpad = 1;

% GUI initiation
h = LearnStatsFig;
handles = guidata(h);

ang_RMSE = 1000; % just to start a while loop

ang_V_RMSE = 1000; % just to start a while loop

tic

while (ang_V_RMSE >= GlobalThreshold)
    
   %Compute and display learning statistics----
    iterations = iterations + 1;
    
    %% Validation Section
    
%Compute the output of hidden neurons for all samples
hid_outmat1 = app_XV * hidneur_weights1;

%Move outputs to the unit circle
hid_outmat1 = hid_outmat1 ./ abs(hid_outmat1);

    app_X1 = [col_app1 hid_outmat1];
    
    %Compute the output of the 2nd hidden neurons for all samples
    hid_outmat2 = app_X1 * hidneur_weights2;
    
    %Move outputs to the unit circle
    hid_outmat2 = hid_outmat2 ./ abs(hid_outmat2);

%append a column of 1s to hid_outmat
hid_outmat2 = [col_app1 hid_outmat2];

%Compute the output of the network
z_outneur = hid_outmat2 * outneur_w;


current_phaseV = mod(angle(z_outneur), 2*pi);

    ang_V_RMSE = 0;
    
    for ii=1:N1
        
        ang_err = abs(y_dV(ii) - current_phaseV(ii));
        
        if (ang_err > pi)
        
            ang_err = 2*pi - ang_err;
        end
        
        ang_V_RMSE = ang_V_RMSE + ang_err^2;
    end

    ang_V_RMSE = sqrt(ang_V_RMSE / N1);  
    
    % Validation_Error will contain validation errors for all iterations
    Validation_Error(iterations) = ang_V_RMSE;
    
    if (ang_V_RMSE < GlobalThreshold) 
        break
    end
    
    %% Regular learning section
    
    %Compute the output of the 1st hidden neurons for all samples
    hid_outmat1 = app_X * hidneur_weights1;
    
    %Move outputs to the unit circle
    abs_hid_outmat1 = abs(hid_outmat1);
    hid_outmat1 = hid_outmat1 ./ abs_hid_outmat1;
    
    % append a column of 1s to X1 from the left
    app_X1 = [col_app hid_outmat1];
    
    % Compute the output of the 2nd hidden neurons for all samples
    hid_outmat2 = app_X1 * hidneur_weights2;
    
    % Move outputs to the unit circle
    abs_hid_outmat2 = abs(hid_outmat2);
    hid_outmat2 = hid_outmat2 ./ abs_hid_outmat2;    
    
    % Calculation of the network error and its backpropagation
    [hid_errmat1, hid_errmat2] = ErrBackProp(hid_outmat2, hidneur_weights2, outneur_w, znet_d, hidneur_num, n, abs_hid_outmat1, abs_hid_outmat2);
    
    % Adjust weights of hidden neurons from the 1st hidden layer
    hidneur_weights1 = HidNeuron_weightadj1(X_pinv, hidneur_weights1, hid_errmat1);
    
    % Compute the output of the 1st hidden neurons for all samples
    hid_outmat1 = app_X * hidneur_weights1;
    
    % Move outputs to the unit circle
    hid_outmat1 = hid_outmat1 ./ abs(hid_outmat1);
    
    % Append a column of 1s to hid_outmat1 from the left
    app_X1 = [col_app hid_outmat1];

    %Pre-compute the SVD of app_X and the pseudo-inverse of app_X. The latter
    %will be used during LLS adjustment of hidden neuron weights.
    %Compute the full SVD of X
    %[U1,S1,V1] = svd(app_X1);

    %Let M = n+1
    %M1 = hidneur_num1+1;

    %Retain only the first M columns of U, and first M rows of S
    %U_hat1 = U1(:, 1:M1);
    %S_hat1 = S1(1:M1, :); %S_hat becomes an M x M square matrix

    %Construct the pseudo-inverse of S
    %S_hpinv1 = diag(1 ./ diag(S_hat1));

    %Construct the pseudo-inverse of X
    %X_pinv1 = V1 * S_hpinv1 * U_hat1';
    
    %Construct the pseudo-inverse matrix of X1
    X_pinv1 = pinv(app_X1);
    
        
    %Adjust weights of hidden neurons from the 2nd hidden layer
    hidneur_weights2 = HidNeuron_weightadj2(X_pinv1, hidneur_weights2, hid_errmat2);
    
    %Compute the output of hidden neurons for all samples
    hid_outmat2 = app_X1 * hidneur_weights2;
    
    %Move outputs to the unit circle
    hid_outmat2 = hid_outmat2 ./ abs(hid_outmat2);
    
    [outneur_w, z_outneur] = OutNeuron_weightadj(hid_outmat2, znet_d, outneur_w);
  
    %error
    err_all = (znet_d - z_outneur./abs(z_outneur))' * (znet_d - z_outneur./abs(z_outneur));

    %Determine the number of errors (nesovpad) and angular RMSE
    current_phase = mod(angle(z_outneur), 2*pi);
    
    nesovpad = 0;
    ang_RMSE = 0;
    
    
    % Flags contin 1 at the positions where current_pase differs from the
    Flags = (current_phase ~= y_d);
    % nesovpad is the number of errors over the entire learning set
    nesovpad = sum(Flags);
    % ang_err angular errors for the entire learning set
    ang_err = abs(y_d - current_phase);
    % Flags contain 1 at the positions where ang_err > pi
    Flags = (ang_err > pi);
    % if ang_err > pi then change it to 2i - ang_err
    ang_err(Flags) = 2*pi - ang_err(Flags);
    % ang_RMSE here is a sume of squared angular errors
    ang_RMSE = ang_RMSE + sum(ang_err.^2);
    % ang_RMSE becomes actual angular RMSE
    ang_RMSE = sqrt(ang_RMSE / N);
    
    % Learning_Error will contain learning errors for all iterations
    Learning_Error(iterations) = ang_RMSE;

    
    %Display the statistic in a separate figure
    set(handles.IterLabel, 'String', num2str(iterations));
    set(handles.ErrLabel, 'String', num2str(err_all));
    set(handles.NesovpadLabel, 'String', [' LError: ' num2str(ang_RMSE)]); %num2str(nesovpad));
    set(handles.AngRMSELabel, 'String', ['VError: ' num2str(ang_V_RMSE) ' LError: ' num2str(ang_RMSE)]);
    
    guidata(h, handles);
    drawnow;
    
    plot(y_dV, 'or');
    hold on
    plot(current_phaseV, '*b');
    hold off   
    
    %Build a list of statistic (for all iterations)
    %verr_all(iterations) = err_all;
    %vang_RMSE(iterations) = ang_RMSE;
    %vnesovpad(iterations) = nesovpad;
    %----
end

toc

close(h);

    figure (3)
    plot(Validation_Error, 'b')
    figure (4)
    plot(Learning_Error, 'r')

disp(' ');
disp(['Iteration: ', num2str(iterations)]);
disp(['Squared norm of error ', num2str(err_all)]);
%disp(['Nesovpad: ', num2str(nesovpad)]);
disp(['Ang L RMSE: ', num2str(ang_RMSE)]);
disp(['Ang V RMSE: ', num2str(ang_V_RMSE)]);

disp('Learning completed!');
end

