function [ydash] = MLP_Test(X, w1, w2, v, d_Per)
% Testing the neural network
% ydash stores the predicted output
% for instances present in matrix X
% w1 stores weights from input layer to hidden layer1
% w2 stores weights from hidden layer1 to hidden layer 2
% v stores weights from hidden layer2 to output layer

% number of examples
N = size(X,1);

% number of hidden layer 1 nodes
H1 = size(w1,2);

% number of hidden layer 2 nodes
H2 = size(w2,2);

% Adding bias term X0 = 1 to the input matrix
X = [ones(N,1) X];
X = (1 - d_Per(1,1)) * X;
% Initializing Hidden Layer 1
Z1 = zeros(N,H1);

% Initializing Hidden Layer 2
Z2 = zeros(N,H2);

% Initializing ydash
ydash = zeros(N,1);

% forward pass to estimate the outputs

% Hidden Layer 1
Z1 = X * w1;
Z1 = sigmf(Z1,[1 0]);

% Hidden Layer 2
% Adding bias term to Hidden Layer 1
Z1 = [ones(N,1) Z1];
Z1 = (1 - d_Per(1,2)) * Z1;
Z2 = Z1 * w2;
Z2 = sigmf(Z2,[1 0]);

% Output Layer
% Adding bias term to hidden layer Z
Z2 = [ones(N,1) Z2];
Z2 = (1 - d_Per(1,3)) * Z2;
ydash = Z2 * v;

end

