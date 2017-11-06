function [ydash] = MLP_Test(X, w1, w2, w3, w4, v)
% Testing the neural network
% ydash stores the predicted output
% for instances present in matrix X
% w1 stores weights from input layer to hidden layer1
% w2 stores weights from hidden layer1 to hidden layer 2
% w3 stores weights from hidden layer2 to hidden layer 3
% v stores weights from hidden layer3 to output layer

% number of examples
N = size(X,1);

% number of hidden layer 1 nodes
H1 = size(w1,2);

% number of hidden layer 2 nodes
H2 = size(w2,2);

% number of hidden layer 3 nodes
H3 = size(w3,2);

% number of hidden layer 4 nodes
H4 = size(w4,2);

% Adding bias term X0 = 1 to the input matrix
X = [ones(N,1) X];

% Initializing Hidden Layer 1
Z1 = zeros(N,H1);

% Initializing Hidden Layer 2
Z2 = zeros(N,H2);

% Initializing Hidden Layer 3
Z3 = zeros(N,H3);

% Initializing Hidden Layer 3
Z4 = zeros(N,H4);

% Initializing ydash
ydash = zeros(N,1);

% forward pass to estimate the outputs

% Hidden Layer 1
Z1 = X * w1;
Z1 = sigmf(Z1,[1 0]);
% Adding bias term to Hidden Layer 1
Z1 = [ones(N,1) Z1];

% Hidden Layer 2
Z2 = Z1 * w2;
Z2 = sigmf(Z2,[1 0]);
% Adding bias term to hidden layer Z
Z2 = [ones(N,1) Z2];

% Hidden Layer 3
Z3 = Z2 * w3;
Z3 = sigmf(Z3,[1 0]);
% Adding bias term to hidden layer 3
Z3 = [ones(N,1) Z3];

% Hidden Layer 4
Z4 = Z3 * w4;
Z4 = sigmf(Z4,[1 0]);
% Adding bias term to hidden layer 3
Z4 = [ones(N,1) Z4];

% Output Layer
ydash = Z4 * v;

end

