function [ang_RMSE, current_phase_tot, y_d] = Net_test(Input, MVN_Data, hidneur_weights1, hidneur_weights2, outneur_w, predictions)

%Determine the number of testing samples and number of inputs
inputs = Input(:,1:47);

y_d = zeros(1, predictions);
current_phase_tot = zeros(1, predictions);


for i = 1:predictions

    y_d(i) = MVN_Data(47+i);
    %Convert input values into complex numbers on the unit circle
    X = exp(1i .* inputs);

    %append a column of 1s to X from the left
    app_X = [1 X];

    %Compute the output of hidden neurons for all samples
    hid_outmat1 = app_X * hidneur_weights1;

    %Move outputs to the unit circle
    hid_outmat1 = hid_outmat1 ./ abs(hid_outmat1);

    app_X1 = [1 hid_outmat1];
    
    %Compute the output of the 2nd hidden neurons for all samples
    hid_outmat2 = app_X1 * hidneur_weights2;
    
    %Move outputs to the unit circle
    hid_outmat2 = hid_outmat2 ./ abs(hid_outmat2);

    %append a column of 1s to hid_outmat
    hid_outmat2 = [1 hid_outmat2];

    %Compute the output of the network
    z_outneur = hid_outmat2 * outneur_w;
    
    current_phase = mod(angle(z_outneur), 2*pi);
    current_phase_tot(i) = current_phase;

    % Update Input array adding actual output at the end
    inputs = [inputs(:, 2:end), current_phase];

end

    ang_RMSE = 0;
    
    for ii=1:predictions
        
        ang_err = abs(y_d(ii) - current_phase_tot(ii));
        
        if (ang_err > pi)
        
            ang_err = 2*pi - ang_err;
        end
        
        ang_RMSE = ang_RMSE + ang_err^2;
    end

    ang_RMSE = sqrt(ang_RMSE / predictions);
    
    plot(y_d, 'or');
    hold on
    plot(current_phase_tot, '*b');
    hold off
    
end