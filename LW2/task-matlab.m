clc; clear;

%% PARAMETERS
RESULTS_DIR = 'results_matlab';
IMG_SIZE = 128;
AUGS_PER_IMAGE = 5;

%% Ensure working directory = script location
cd(fileparts(mfilename('fullpath')));

%% Reset results folder
if exist(RESULTS_DIR, 'dir')
    rmdir(RESULTS_DIR, 's');
end
mkdir(RESULTS_DIR);

%% Check dataset folder
datasetPath = fullfile(pwd, 'dataset');
if ~exist(datasetPath, 'dir')
    error("'dataset' folder not found in current directory.");
end

%% Get all images recursively
imageFiles = dir(fullfile(datasetPath, '**', '*.*'));
imageFiles = imageFiles(~[imageFiles.isdir]);

validExt = {'.jpg','.jpeg','.png'};

%% PROCESS LOOP
for k = 1:length(imageFiles)

    [~, base, ext] = fileparts(imageFiles(k).name);

    if ~ismember(lower(ext), validExt)
        continue;
    end

    imgPath = fullfile(imageFiles(k).folder, imageFiles(k).name);
    img = imread(imgPath);

    if size(img,3) ~= 3
        continue; % skip non-RGB images
    end

    %% Resize
    img = imresize(img, [IMG_SIZE IMG_SIZE]);

    %% Color Conversions
    gray = rgb2gray(img);
    hsv = rgb2hsv(img);
    lab = rgb2lab(img);

    imwrite(gray, fullfile(RESULTS_DIR, base + "_gray.png"));
    imwrite(hsv2rgb(hsv), fullfile(RESULTS_DIR, base + "_hsv.png"));
    imwrite(lab2rgb(lab), fullfile(RESULTS_DIR, base + "_lab.png"));

    %% AUGMENTATIONS
    for i = 0:AUGS_PER_IMAGE-1

        aug = img;

        % --- Random Rotation (-30 to 30)
        angle = -30 + 60 * rand();
        aug = imrotate(aug, angle, 'bilinear', 'crop');

        % --- Random Flip
        flipChoice = randi(3);
        if flipChoice == 1
            aug = flipud(aug);
        elseif flipChoice == 2
            aug = fliplr(aug);
        else
            aug = flipud(fliplr(aug));
        end

        % --- Random Brightness (0.7–1.3)
        factor = 0.7 + 0.6 * rand();
        aug = im2double(aug);
        aug = aug * factor;
        aug = im2uint8(mat2gray(aug));

        % --- Median Blur (3 or 5 kernel)
        if rand() < 0.5
            ksize = 3;
        else
            ksize = 5;
        end
        augGray = medfilt2(rgb2gray(aug), [ksize ksize]);
        aug = repmat(augGray, [1 1 3]);

        % --- Bilateral Filter
        aug = imbilatfilt(aug);

        % --- Gaussian Noise (approx σ=15)
        aug = imnoise(aug, 'gaussian', 0, (15/255)^2);

        % Save augmented image
        filename = sprintf('%s_aug%d.png', base, i);
        imwrite(aug, fullfile(RESULTS_DIR, filename));
    end
end

disp('Processing completed successfully.');