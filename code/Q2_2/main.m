% Check if the weights.mat file exists, 
% otherwise train to create them
disp('Loading/ Creating weights');
if ( exist('weights1.mat', 'file') == 0 )
    main1;
end
W1 = load('weights1.mat');

if ( exist('weights2.mat', 'file') == 0 )
    main2;
end
W2 = load('weights2.mat');

if ( exist('weights3.mat', 'file') == 0 )
    main3;
end
W3 = load('weights3.mat');

% Loading the training data if feature file does not exist

% Parameters for the global descriptor
block_size = 8;
no_blocks = 49; %There will be 50% overlap between the blocks
gradient_bins = 9;

if ( exist('features.csv', 'file') == 0 )

    pos_path = '../steering/';

    fileID = fopen('../steering/data.txt','r');
    A = textscan(fileID,'%c %c %s %f');
    pos_imgfiles = A{3}; % ignoring first 2 characters
    fclose(fileID);
    
    disp('Creating features');
    
    pos_imgfiles = char(pos_imgfiles);
    no_images = length(pos_imgfiles)-1;

    all_image_features = zeros(no_images, (no_blocks*gradient_bins));
    for i = 1:no_images
        if (rem(i,1000)==1)
            fprintf('%dth image\n', i);
        end
        ith_image = rgb2gray(imread(strcat(pos_path, pos_imgfiles(i+1,:))));
        for j = 0:no_blocks-1
            row = floor(j/sqrt(no_blocks));
            col = rem(j,sqrt(no_blocks));
            patch_image = ith_image(row*(block_size/2)+1:row*(block_size/2)+block_size, col*(block_size/2)+1:col*(block_size/2)+block_size);
            patch_descriptor = ComputePatchDescriptor(patch_image, gradient_bins);
            all_image_features(i, j*gradient_bins+1:((j+1)*gradient_bins)) = patch_descriptor;
        end
    end
    csvwrite('features.csv',all_image_features);
end

Y = csvread('features.csv');



% Filename containing the features of the test images
feature_filename = 'features_test.csv';

% Parameters for the global descriptor
block_size = 8;
no_blocks = 49; %There will be 50% overlap between the blocks
gradient_bins = 9;


disp('Loading/ Creating features from images');
if ( exist(feature_filename, 'file') == 0 )

    % Opening the test data
    pos_path = '../l3-test/';

    fileID = fopen('../l3-test/test-data.txt','r');
    A = textscan(fileID,'%c %c %s %f');
    pos_imgfiles = A{3}; % ignoring first 2 characters
    fclose(fileID);

    disp('Creating features');
    
    pos_imgfiles = char(pos_imgfiles);
    no_images = length(pos_imgfiles);

    all_image_features = zeros(no_images, (no_blocks*gradient_bins));
    for i = 1:no_images
        if (rem(i,1000)==1)
            fprintf('%dth image\n', i);
        end
        ith_image = rgb2gray(imread(strcat(pos_path, pos_imgfiles(i,:))));
        for j = 0:no_blocks-1
            row = floor(j/sqrt(no_blocks));
            col = rem(j,sqrt(no_blocks));
            patch_image = ith_image(row*(block_size/2)+1:row*(block_size/2)+block_size, col*(block_size/2)+1:col*(block_size/2)+block_size);
            patch_descriptor = ComputePatchDescriptor(patch_image, gradient_bins);
            all_image_features(i, j*gradient_bins+1:((j+1)*gradient_bins)) = patch_descriptor;
        end
    end
    csvwrite(feature_filename,all_image_features);
end

X = csvread(feature_filename);

% Normalizing the images
X = (X - meshgrid(mean(Y),1:size(X,1)))./meshgrid(std(Y),1:size(X,1));  

o1 =  MLP_Test(X, W1.w11, W1.w21, W1.w31, W1.w41, W1.v1);
o2 =  MLP_Test(X, W2.w12, W2.w22, W2.w32, W2.w42, W2.v2);
o3 =  MLP_Test(X, W3.w1, W3.w2, W3.w3, W3.w4, W3.v);

o = (o1+o2+(o3.*2))./4;

dlmwrite('test_output.txt', o);