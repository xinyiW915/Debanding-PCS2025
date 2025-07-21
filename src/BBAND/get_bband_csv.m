% Select dataset folder
folderPath = uigetdir;  % Open dialog to select folder with images
imageFiles = dir(fullfile(folderPath, '*.png')); % Change to match your image format

% Set batch size
batch_size = 500;
total_images = length(imageFiles);
num_batches = ceil(total_images / batch_size);

% Initialize results cell array
results = cell(total_images + 1, 2); 
results(1, :) = {'ImageName', 'BandingScore'}; % Add header for CSV

% Set BBAD_I function parameters
imguidedfilt_ws = 1; % Default value, change if needed
thr1 = 2;  % Default threshold
thr2 = 12; % Default threshold

% Process each image (remove batch logic, process all at once)
for i = 1:total_images
    % Read image
    imagePath = fullfile(folderPath, imageFiles(i).name);
    img = imread(imagePath);
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Run BBAD_I function on the image
    [~, ~, band_score, ~, ~] = BBAD_I(img, imguidedfilt_ws, thr1, thr2);
    
    % Store image name and score
    results{i + 1, 1} = imageFiles(i).name;
    results{i + 1, 2} = band_score;
    
    fprintf('Processed %s -> Score: %.4f\n', imageFiles(i).name, band_score);
end

% Convert results to table and save as a single CSV file
resultsTable = cell2table(results(2:end, :), 'VariableNames', results(1, :));

% Extract folder name for output file naming
[~, folderName] = fileparts(folderPath);
outputFileName = sprintf('banding_scores_%s.csv', folderName);
outputFilePath = fullfile(pwd, outputFileName);  % Save to current working directory

writetable(resultsTable, outputFilePath);

fprintf('All images processed. Results saved to: %s\n', outputFilePath);
