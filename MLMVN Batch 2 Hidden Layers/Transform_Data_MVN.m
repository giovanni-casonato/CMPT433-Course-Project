function [Data] = Transform_Data_MVN(Input, transformed_data)

min_value = min(Input) * 0.85;
max_value = max(Input) * 1.15;

reverse = true; % Change to false to get (0, 2pi) intervals. true to get original data.

if reverse == false
    tot_values = numel(Input);
    Data = zeros(size(Input));

    for i = 1:tot_values
        original_value = Input(i);
        transformed_value = (original_value - min_value) / (max_value - min_value) * 2*pi;
        Data(i) = transformed_value;
    end

else
    tot_values = numel(transformed_data);
    Data = zeros(size(transformed_data));

    for ii = 1:tot_values
        transformed_value = transformed_data(ii);
        original_value = (transformed_value / (2*pi)) * (max_value - min_value) + min_value;
        Data(ii) = original_value;
    end

end

