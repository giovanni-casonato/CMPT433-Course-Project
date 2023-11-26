function [RMSE] = TestingMLP(Input, net)
% This program performs test of a trained MLP-MLF with 2 hidden layers
% It takes inputs with desired outputs, calculates actual outputs and RMSE
% between actual and desired outputs

% Input contains a learning set (last column is a desired output)
% net is a cell array of weights

%Extraction of weights
WeightsH1 = net{1};
WeightsH2 = net{2};
WeightsOut = net{3};

[N,ninputs]=size(Input);

% Desired outputs
targets=Input(:,ninputs);

ninputs=ninputs-1;

% inputs contain only inputs
inputs = Input(: , 1:ninputs);

% An array to store actual outputs after learning
ActualOutputs = zeros(1, N);

% N is now the number of learning samples
% ninputs is now the number of inputs

% a for loop over all learning samples
for j=1:N
         % calculation of the actual output of the network for the j-th
         % sample
         output  = EvalNN( inputs(j,:), WeightsH1, WeightsH2, WeightsOut );
         % Accumulation of actual outputs
         ActualOutputs(j) = output;
         disp([' Inputs [',num2str(inputs(j,:)), ' ] --> Actual Output:  [',num2str(output),']', '] --> Desired Output:  [',num2str(targets(j)),']'] )
         
end
% MSE over all testing samples
error = sum((ActualOutputs - targets').^2)/N;
% RMSE
RMSE = sqrt(error);
fprintf('Prediction/Recognition Error = %f \n', RMSE);
figure (2);
hold off
plot(targets,'Or'); 
hold on
plot(ActualOutputs, '*b');
end

