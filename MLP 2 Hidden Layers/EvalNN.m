function [ Output ] = EvalNN( inputs, WeightsH1, WeightsH2, WeightsOut)

% A Matlab implementation of MLF with
% backpropagation training with momentum.
% Demo
% 

%input activations
inputs     = [1 inputs]'; % 1 for bias node;

% 1st hidden layer
z1    = WeightsH1 * inputs;
y1     = tanh(z1);

% 2nd hidden layer

y1 = [1; y1];
z2    = WeightsH2 * y1;
y2     = tanh(z2);

% output activations
y2 = [1; y2];
z3   =  WeightsOut * y2;
Output = tanh(z3);

