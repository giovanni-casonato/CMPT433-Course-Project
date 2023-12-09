% Demo of time series prediction using MLP with 2 hidden layers
% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 

k = 96;
n = 48;
Input = MLP_Data;

learningSetMLP = zeros(k, n);

for i = 1:k
    input_sequence = Input(i: n + i - 1);

    if i + n <= numel(Input)
        output_element = Input(i + n - 1);

        learningSetMLP(i, 1:n) = input_sequence;
        learningSetMLP(i, end) = output_element;
    end
end

[Network, RMSE] = LearningMLP(learningSetMLP, [8 90], 0.05, 3000);
% Network - weights
% RMSE - resulting learning RMSE
pause(1)


% Testing

testingSetMLP = zeros(1, 47);
input_sequence = MLP_Data(49:95); testingSetMLP(1, 1:47) = input_sequence;

[Actual_Outputs] = TestingMLP(testingSetMLP, Network, MLP_Data, 48);



