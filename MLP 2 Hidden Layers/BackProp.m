function [ WeightsH1, WeightsH2, WeightsOut ] = BackProp(inputs,target,ninputs,network,WeightsH1, WeightsH2, WeightsOut)
% A Matlab implementation of MLP-MLF with backpropagation learning with
% WeightsH1, WeightsH2, WeightsOut - arrays with all weights
% inputs - array with inputs
% targets - array with desired outputs
% ninputs - # of inputs
% network is a [n1 n2] vecor containing # of hidden neurons in layers
% 
% 

%input activations
inputs     = [1 inputs]'; % 1 for bias node;

% 1st hidden layer
z1    = WeightsH1 * inputs;
y1     = tanh(z1);

% 2nd hidden layer

y1r = [1; y1];
z2    = WeightsH2 * y1r;
y2     = tanh(z2);

% output activations
y2r = [1; y2];
z3   =  WeightsOut * y2r;
Output = tanh(z3);

% calculate errors for the output neurons
% (1.0-ao.^2) is a derivative of the tanh activation function
output_errors = (1.0-Output.^2) * (target - Output);

% backpropagation of the output neurons' errors to the 2nd hidden layer neurons
error = output_errors * WeightsOut(2:end)';
% (1.0-y2^.^2) is a derivative of the tanh activation function
hidden_errors2 = (1.0-y2.^2) .* error;

% backpropagation of the 2nd neurons' errors to the 1st hidden layer neurons
error = WeightsH2(:,2:end)' *  hidden_errors2;
% (1.0-ah.^2) is a derivative of the tanh activation function
hidden_errors1 = (1.0-y1.^2) .* error;

% update 1st hidden layer neurons' weights
change  = hidden_errors1 .* inputs'; 
WeightsH1 = WeightsH1 + (1/(ninputs+1)) .* change;

% update 1st hidden layer neurons' outputs
% weighted sums for all hidden neurons
z1    = WeightsH1 * inputs;
y1     = tanh(z1);

y1r = [1; y1];

% update 2nd hidden layer neurons' weights
change  = hidden_errors2 .* y1r'; 
WeightsH2 = WeightsH2 + 1/(network(1)+1) .* change;

% update 2nd hidden layer neurons' outputs
% weighted sums for all hidden neurons
% 2nd hidden layer
y1r = [1; y1];
z2    = WeightsH2 * y1r;
y2     = tanh(z2);

y2r = [1; y2];
% update output neurons' weights
change  = output_errors * y2r;
WeightsOut = WeightsOut + 1/(network(2)+1) .* change';


end

