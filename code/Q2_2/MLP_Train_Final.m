function [bestw1, bestw2, bestw3, bestw4, bestv, train_error, val_error] = MLP_Train_Final(X, Y, X_val, Y_val, nEpochs, eta, mini_Bsize, d_per)
% Training the neural network
% w1 stores weights from input layer to hidden layer1
% w2 stores weights from hidden layer1 to hidden layer 2
% v stores weights from hidden layer2 to output layer
% X - training data of size NxD
% Y - training labels
% nEpochs - the number of training epochs
% eta - the learning rate
% mini_Bsize - the mini-batch size
% d_Per - Dropout percentage

alpha = 0.0001;

% number of training data points
N = size(X,1);
% number of features
D = size(X,2); % excluding the bias term
% number of nodes in hidden layer 1
H1 = 1024;
% number of nodes in hidden layer 2
H2 = 512;
% number of nodes in hidden layer 3
H3 = 256;
% number of nodes in hidden layer 4
H4 = 64;

% weights for the connections between different layers
% random values from the interval [-0.1 0.1]

A = 1/sqrt(D);
% w1 is a (D+1) x H1 matrix
w1 = normrnd(0,A,[D+1 H1]);

% w2 is (H1+1) x H2 matrix
w2 = normrnd(0,A,[H1+1 H2]);

% w3 is (H2+1) x H3 matrix
w3 = normrnd(0,A,[H2+1 H3]);

% w4 is (H3+1) x H4 matrix
w4 = normrnd(0,A,[H3+1 H4]);

% v is a (H4+1) x 1 matrix
v = normrnd(0,A,[H4+1 1]);

% Adding bias term X0 = 1 to the input matrix
X1 = [ones(N,1) X];

der5 = zeros(D+1,H1);
der4 = zeros(H1+1,H2);
der3 = zeros(H2+1,H3);
der2 = zeros(H3+1,H4);
der1 = zeros(H4+1,1);
a = 0.001;
b = 0.5;
%lambda = 10;
minerror = 1000000;


% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    % Deciding nodes to dropout in each batch
    d1 = randperm(D)+1;
    dropout_X = d1(1:round(d_per(1,1)*(D)));
    d2 = randperm(H1)+1;
    dropout_Z1 = d2(1:round(d_per(2,1)*(H1)));
    d3 = randperm(H2)+1;
    dropout_Z2 = d3(1:round(d_per(3,1)*(H2)));
    d4 = randperm(H3)+1;
    dropout_Z3 = d4(1:round(d_per(4,1)*(H3)));
    d5 = randperm(H4)+1;
    dropout_Z4 = d5(1:round(d_per(5,1)*(H4)));
    
    for n = 1:mini_Bsize:N
        %% forward pass - calculating the output of the hidden layer units
        
        % the current training point is X(iporder(n), :)
        E = min(N, n+mini_Bsize-1);
        siz = E - n + 1;
        X_train = X1(n:E, :);
        % Dropout from input layer
        X_train(:,dropout_X) = 0;
        
        % Hidden Layer 1
        
        z = X_train * w1;
        Z1 = sigmf(z,[1,0]);
        % Adding bias term to Z1
        Z1 = [ones(size(Z1,1),1) Z1];
        % Dropout from Hidden Layer 1
        Z1(:,dropout_Z1) = 0;
        % Normalizing Hidden Layer 1
        %Z1 = (Z1 - meshgrid(mean(Z1),1:size(Z1,1)))./meshgrid(std(Z1),1:size(Z1,1));   

        % Hidden Layer 2
        
        z = Z1 * w2;
        Z2 = sigmf(z,[1,0]);
        % Adding bias term to hidden layer 2
        Z2 = [ones(size(Z2,1),1) Z2];
        % Dropout from Hidden Layer 2
        Z2(:,dropout_Z2) = 0;
        % Normalizing Hidden Layer 2
        %Z2 = (Z2 - meshgrid(mean(Z2),1:size(Z2,1)))./meshgrid(std(Z2),1:size(Z2,1));   

        % Hidden Layer 3
        
        z = Z2 * w3;
        Z3 = sigmf(z,[1,0]);
        % Adding bias term to hidden layer 2
        Z3 = [ones(size(Z3,1),1) Z3];
        % Dropout from Hidden Layer 2
        Z3(:,dropout_Z3) = 0;
        % Normalizing Hidden Layer 3
        %Z3 = (Z3 - meshgrid(mean(Z3),1:size(Z3,1)))./meshgrid(std(Z3),1:size(Z3,1));   

        % Hidden Layer 4
        
        z = Z3 * w4;
        Z4 = sigmf(z,[1,0]);
        % Adding bias term to hidden layer 2
        Z4 = [ones(size(Z4,1),1) Z4];
        % Dropout from Hidden Layer 2
        Z4(:,dropout_Z4) = 0;
        
        % Hidden to Output layer
        
        O = Z4 * v;
        
        %% backward pass - updating weights between connections
        
        % outlayer units
        
        % Computing Gradient
        del1 = O - Y(n:E, : );
        % Updating weight
        v = v - (eta/siz)*Z4'*del1 + (alpha * der1);
        
        der1 = (eta/siz)*Z4'*del1;
        
        % hidden layer unit 4
        
        % Computing gradient
        del2 = del1 * v' .* Z4 .* (1-Z4);
        del2 = del2(:, 2:size(del2,2));
        % Updating weight
        w4 = w4 - (eta/siz)*Z3'*del2 + (alpha * der2);
        der2 = (eta/siz)*Z3'*del2;
        
        % hidden layer unit 3
        
        % Computing gradient
        del3 = del2 * w4' .* Z3 .* (1-Z3);
        del3 = del3(:, 2:size(del3,2));
        % Updating weight
        w3 = w3 - (eta/siz)*Z2'*del3 + (alpha * der3);
        der3 = (eta/siz)*Z2'*del3;
        
        % hidden layer unit 2
        
        % Computing gradient
        del4 = del3 * w3' .* Z2 .* (1-Z2);
        del4 = del4(:, 2:size(del4,2));
        % Updating weight
        w2 = w2 - (eta/siz)*Z1'*del4 + (alpha * der4);
        der4 = (eta/siz)*Z1'*del4;
        
        % hidden layer unit 1
        
        % Computing gradient
        del5 = del4 * w2' .* Z1 .* (1-Z1);
        del5 = del5(:, 2:size(del5,2));
        % Updating weight
        
        w1 = w1 - (eta/siz)*X_train'*del5 + (alpha * der5);
        der5 = (eta/siz)*X_train'*del5;
        
    end
    %% Computing the training and validation error
    
    ydash = MLP_Test(X, w1, w2, w3, w4, v);
    yydash = MLP_Test(X_val, w1, w2, w3, w4, v);
    train_error(epoch) = sum((Y - ydash).^2)/size(Y,1);
    val_error(epoch) = sum((Y_val - yydash).^2)/size(Y_val,1);
    % displaying error after each epoch
    disp(sprintf('training error after epoch %d: %f\n',epoch,...
        train_error(epoch)));
    disp(sprintf('validation error after epoch %d: %f\n',epoch,...
        val_error(epoch)));
    
    % Saving the weights
    if (val_error(epoch)<minerror)
        bestw1 = w1;
        bestw2 = w2;
        bestw3 = w3;
        bestw4 = w4;
        bestv = v;
        minerror = val_error(epoch);
    end
    %% Adaptive Learning Rate
    if(epoch > 1 && val_error(epoch) < val_error(epoch - 1))
        eta = eta + a;
    else
        eta = eta - b*eta;
    end
end
 
end

