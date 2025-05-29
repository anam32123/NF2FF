function output_data=exclude_7sigma_outliers(data)
    % finds all outliers greater than 7 standard deviations away from the
    % mean and sets them to NaN values in the dataset
    
    output_data=data;

    sigma_data=std(data,0,'all');
    preliminary_mean=mean(data,'all');
    outliers_indices=find(abs(data-preliminary_mean) > (7*sigma_data));
    output_data(outliers_indices)=NaN;