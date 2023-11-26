% Demo of time series prediction using MLMVN with 2 hidden layers
% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively
[hidneur_weights1, hidneur_weights2, outneur_w, iterations] = Net_learn(Progr_Learning_MVN, [8 32], 0.005);
% hidneur_weights1 - weights of the 1st hidden layer
% hidneur_weights2 - weights of the 2nd hidden layer
% weights of the output neurons
% iterations - # of iterations

pause(3)

% Testing of the trained network
[ang_RMSE, actual_output, desired_output] = Net_test(Progr_Testing_MVN, hidneur_weights1, hidneur_weights2, outneur_w)
% ang_RMSE - resulting angular RMSE
 