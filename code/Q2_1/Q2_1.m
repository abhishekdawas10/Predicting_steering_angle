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
fprintf('Reading Images!!!\n');
if(exist('Images.csv','file')==0)
    pos_imgfiles = char(pos_imgfiles);
    X = zeros(length(pos_imgfiles)-1,1024);
    for i = 2:length(pos_imgfiles)
        I = imread(strcat(pos_path, pos_imgfiles(i,:)));
        X(i-1,:) = reshape(rgb2gray(I)',[1 1024]);
    end
    csvwrite('Images.csv',X);
end
X = csvread('Images.csv');

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

fprintf('Data partitioned into training and validation!!!\n');
%% Inputting useful parameters for network

% number of epochs for training
prompt = 'Enter the number of trainig epochs - ';
nEpochs = input(prompt);

% learning rate
prompt = 'Enter the learning rate - ';
eta = input(prompt);

% mini batch size
prompt = 'Enter mini batch size - ';
mini_Bsize = input(prompt);


% dropout percentage
prompt = 'Enter dropout percentage for Layer 1 - ';
d_Per(1,1) = input(prompt);
prompt = 'Enter dropout percentage for Layer 2 - ';
d_Per(1,2) = input(prompt);
prompt = 'Enter dropout percentage for Layer 3 - ';
d_Per(1,3) = input(prompt);

%% Training the neural network
% w1 stores weights from input layer to hidden layer1
% w2 stores weights from hidden layer1 to hidden layer 2
% v stores weights from hidden layer2 to output layer
[w1, w2, v, train_error, val_error] = MLP_Train(X_train, Y_train, X_val, Y_val, nEpochs, eta, mini_Bsize, d_Per);
fig1 = figure;
plot(1:nEpochs, train_error, 1:nEpochs, val_error);
xlabel('Number of Epochs');
ylabel('Error');
legend('Training Set Error', 'Validation Set Error'); 

  

