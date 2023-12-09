function [ActualOutputs] = TestingMLP(Input, net, Data_MLP, predictions)
    % This program performs test of a trained MLP-MLF with 2 hidden layers
    % It takes inputs with desired outputs, calculates actual outputs and RMSE
    % between actual and desired outputs

    % Input contains a learning set (last column is a desired output)
    % net is a cell array of weights

    % Extraction of weights
    WeightsH1 = net{1};
    WeightsH2 = net{2};
    WeightsOut = net{3};

    ActualOutputs = zeros(1, predictions);
    targets = zeros(1, predictions);
    inputs = Input(1, 1:47);

    for i = 1:predictions
        targets(i) = Data_MLP(96+i); % Desired output
         % Input values excluding the desired output
        
        % Calculation of the actual output of the network
        output = EvalNN(inputs, WeightsH1, WeightsH2, WeightsOut);
        ActualOutputs(i) = output;

        disp([' Inputs [', num2str(inputs), ' ] --> Actual Output:  [', num2str(output), '] --> Desired Output:  [', num2str(targets(i)), ']'])
        
        % Update Input array adding actual output at the end
        inputs = [inputs(2:end), output];
    end

% MSE over all testing samples
error = sum((ActualOutputs - targets).^2)/predictions;
% RMSE
RMSE = sqrt(error);
fprintf('Prediction/Recognition Error = %f \n', RMSE);
figure (2);
hold off
plot(targets,'Or'); 
hold on
plot(ActualOutputs, '*b');
end
