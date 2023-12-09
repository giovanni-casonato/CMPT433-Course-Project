% Demo of time series prediction using MLMVN with 2 hidden layers
% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively

k = 96;
n = 48;
Input = MVN_Data;

learningSetMVN = zeros(k, n);

for i = 1:k
    input_sequence = Input(i: n + i - 1);

    if i + n <= numel(Input)
        output_element = Input(i + n - 1);

        learningSetMVN(i, 1:n) = input_sequence;
        learningSetMVN(i, end) = output_element;
    end
end


[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(learningSetMVN, [8 200], 0.005);
% hidneur_weights1 - weights of the 1st hidden layer
% hidneur_weights2 - weights of the 2nd hidden layer
% weights of the output neurons
% iterations - # of iterations

testingSetMVN = zeros(1, 47);
input_sequence = MVN_Data(49:95); testingSetMVN(1, 1:47) = input_sequence;

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(testingSetMVN, MVN_Data, hidneur_weights1, hidneur_weights2, outneur_w, 48);
% ang_RMSE - resulting angular RMSE


 