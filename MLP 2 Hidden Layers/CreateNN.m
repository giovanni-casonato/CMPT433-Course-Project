function [ net ] = CreateNN( ninputs, network )
% A Matlab implementation of a MultiLayer perceptron with
% backpropagation training with momentum.
% 
% 

rng('shuffle');
% initial weights for the 1st hidden layer
WeightsH1 = rand(network(1), ninputs+1) - 0.5;
% initial weights for the 2nd hidden layer
WeightsH2 = rand(network(2), network(1)+1) - 0.5;
% initital weights for the output neuron
WeightsOut = rand(1, network(2)+1) - 0.5;

net{1} = WeightsH1;
net{2} = WeightsH2;
net{3} = WeightsOut;

end
