% Neural Network to predict steering
% angle from road image

close all;
clear all;
clc;

%% Loading Data
%  The matrix X stores 32 X 32 image as a single row
%  X is of size N X 1024
%  The matrix Y stores the label for N trainig examples

pos_path = '../steering/';

fileID = fopen('../steering/data.txt','r');
A = textscan(fileID,'%c %c %s %f');
 
pos_imgfiles = A{3}; % ignoring first 2 characters

labels = A{4}; 
Y = labels(2:size(labels,1),:); % Ignoring label for img0
  
fclose(fileID);


% Filename containing the features of the images
feature_filename = 'features.csv';

% Parameters for the global descriptor
block_size = 8;
no_blocks = 49; %There will be 50% overlap between the blocks
gradient_bins = 9;


if ( exist(feature_filename, 'file') == 0 )
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
    csvwrite(feature_filename,all_image_features);
end


X = csvread(feature_filename);

% Normalizing the images
X = (X - meshgrid(mean(X),1:size(X,1)))./meshgrid(std(X),1:size(X,1));  

fprintf('Images Read!!!\n');
%% Splitting data into 80% Training and 20% Validation
%  X_train contains training examples
%  X_val contains validation examples

% Randomizing the order of Input data
order = randperm(size(X,1));
X = X(order,:);
Y = Y(order,:);

div = round(0.8*size(X,1));

X_train = X(1:div,:);
Y_train = Y(1:div,:);
X_val = X(div+1:size(X,1),:);
Y_val = Y(div+1:size(Y,1),:);

fprintf('Data partitioned!!!\n');
%% Inputting useful parameters for network

% number of epochs for training
nEpochs = 5000;

% learning rate
eta = 0.01;

% mini batch size
mini_Bsize = 64;

% dropout percentage
d_Per = [0; 0; 0; 0; 0];

%% Training the neural network
% w1 stores weights from input layer to hidden layer1
% w2 stores weights from hidden layer1 to hidden layer 2
% v stores weights from hidden layer2 to output layer
[w1, w2, w3, w4, v, train_error, val_error] = MLP_Train_Final(X_train, Y_train, X_val, Y_val, nEpochs, eta, mini_Bsize, d_Per);
save('weights3.mat', 'w1', 'w2', 'w3', 'w4', 'v');
fprintf('Minimum Validation Error = ');
disp(min(val_error));
