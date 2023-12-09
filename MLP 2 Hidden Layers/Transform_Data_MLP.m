function [Data] = Transform_Data_MLP(Input, transformed_data)

min_value = min(Input) * 0.85;
max_value = max(Input) * 1.15;

reverse = true; % Change to false to get (-1, 1) intervals. true to get original data.

if reverse == false
    tot_values = numel(Input);
    Data = zeros(size(Input));

    for i = 1:tot_values
        original_value = Input(i);
        transformed_value = 2 * (original_value - min_value) / (max_value - min_value) - 1;
        Data(i) = transformed_value;
    end

else
    tot_values = numel(transformed_data);
    Data = zeros(size(transformed_data));

    for i = 1:tot_values
        transformed_value = transformed_data(i);
        original_value = ((transformed_value + 1) / 2) * (max_value - min_value) + min_value;
        Data(i) = original_value;
    end

end

