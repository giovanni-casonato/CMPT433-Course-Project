function [net, RMSE] = LerningMLP(Input, network, rmseThreshold, iterationsLimit )
% A Matlab implementation of MLP-MLF with 2 hidden layers and 1 output
% neuron
% Input contains a learning set (last column is a desired output)
% network is a [n1 n2] vecor containing # of hidden neurons in layers

% rmseThreshold - threshold for RMSE (learning target)
% iterationsLimit - max # of iterations

%clc
%clear all
%close all


OutputFlag = 1;

[N,ninputs]=size(Input);

% Desired outputs
targets=Input(:,ninputs);

ninputs=ninputs-1;

% inputs contain only inputs
inputs = Input(: , 1:ninputs);

% An array to store actual outputs after learning
ActualOutputs = zeros(1, N); 

net=CreateNN(ninputs,network);

WeightsH1 = net{1};
WeightsH2 = net{2};
WeightsOut = net{3};

% Set initial error to start the while loop with learning
RMSE=10;
% Set a counter of iterations
iteration=0;

% A main loop with the learning process
while (iteration<=iterationsLimit)&&(RMSE>rmseThreshold)
% learning continues as long as error>errorlimit and iteration<=iterationslimit  
    % increment intreations
    iteration=iteration+1;
    
    % Evaluation of RMSE for the entire learning set
    % a for loop over all learning samples
    for j=1:N
         % calculation of the actual output of the network for the j-th
         % sample
         output  = EvalNN(inputs(j,:), WeightsH1, WeightsH2, WeightsOut );
         % Accumulation of actual outputs
         ActualOutputs(j) = output;
    end
    % MSE over all learning samples
    error = sum((ActualOutputs - targets').^2)/N;
    % RMSE
    RMSE = sqrt(error);
    
    if mod(iteration,OutputFlag)==0
        display([' Iteration ', num2str(iteration), '  ' 'RMSE = ',num2str(RMSE)])
    end
    
    % if RMSE dropped below errorlimit, then the learning process converged
    if RMSE <= rmseThreshold
        break  % and we get out of the while loop
    end
        
    % otherwise we start correction of the weights
        
    % a for loop over all learning samples
    for j=1:N
         % backpropagation and correction of the weights
         [ WeightsH1, WeightsH2, WeightsOut ] = BackProp(inputs(j,:),targets(j),ninputs,network, WeightsH1, WeightsH2, WeightsOut);
      
    end
   
end

net{1} = WeightsH1;
net{2} = WeightsH2;
net{3} = WeightsOut;

%test newtork function
% final results of the learning process
fprintf(' Iterations = %7d \n',iteration); 
for j=1:N
         output  = EvalNN( inputs(j,:), WeightsH1, WeightsH2, WeightsOut  );
         ActualOutputs(j) = output;
         disp([' Inputs [',num2str(inputs(j,:)),'] --> outputs:  [',num2str(output),']']) 
         
end
display(['Error= ',num2str(error)]);
figure (1);
hold off
plot(targets,'or'); 
hold on
plot(ActualOutputs, '*g');
    

end

