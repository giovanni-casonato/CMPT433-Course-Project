% Demo of time series prediction using MLMVN with 2 hidden layers
% Learning using a network with 2 hidden layers [n k] containing n and k
% neurons, respectively. 
[Network, RMSE] = LerningMLP(LearnProgr_MLP, [8 32], 0.005, 3000 );
% Network - weights
% RMSE - resulting learning RMSE

pause(3)

% Testing of the trained network

 [RMSE] = TestingMLP(TestProgr_MLP, Network)