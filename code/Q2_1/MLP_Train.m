function [w1, w2, v, train_error, val_error] = MLP_Train(X, Y, X_val, Y_val, nEpochs, eta, mini_Bsize, d_Per)
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


% number of training data points
N = size(X,1);
% number of features
D = size(X,2); % excluding the bias term
% number of nodes in hidden layer 1
H1 = 512;
% number of nodes in hidden layer 2
H2 = 64;

% weights for the connections between different layers
% random values from the interval [-0.1 0.1]

% w1 is a (D+1) x H1 matrix
w1 = -0.01 + (0.02)*rand(H1,D);
w1 = [zeros(H1,1) w1];
w1 = w1';

% w2 is (H1+1) x H2 matrix
w2 = -0.01 + (0.02)*rand(H2,H1);
w2 = [zeros(H2,1) w2];
w2 = w2';

% v is a (H2+1) x 1 matrix
v = -0.01 + (0.02)*rand(1,H2);
v = [zeros(1,1) v];
v = v';

% Adding bias term X0 = 1 to the input matrix
X1 = [ones(N,1) X];

% mlp training through stochastic gradient descent
for epoch = 1:nEpochs
    % Deciding nodes to dropout in each batch
    rand1 = randperm(D)+1;
    dropout_X = rand1(1:round(d_Per(1,1)*(D)));
    rand2 = randperm(H1)+1;
    dropout_Z1 = rand2(1:round(d_Per(1,2)*(H1)));
    rand3 = randperm(64)+1;
    dropout_Z2 = rand3(1:round(d_Per(1,3)*(H2)));
    
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
        
        % Hidden Layer 2
        
        z = Z1 * w2;
        Z2 = sigmf(z,[1,0]);
        % Adding bias term to hidden layer 2
        Z2 = [ones(size(Z2,1),1) Z2];
        % Dropout from Hidden Layer 2
        Z2(:,dropout_Z2) = 0;
        
        % Hidden to Output layer
        
        O = Z2 * v;
        
        %% backward pass - updating weights between connections
        
        % outlayer units
        
        % Computing Gradient
        del1 = O - Y(n:E, : );
        % Updating weight
        v = v - (eta/siz)*Z2'*del1;
        
        % hidden layer unit 2
        
        % Computing gradient
        del2 = del1 * v' .* Z2 .* (1-Z2);
        del2 = del2(:, 2:size(del2,2));
        % Updating weight
        w2 = w2 - (eta/siz)*Z1'*del2;
        
        % hidden layer unit 1
        
        % Computing gradient
        del3 = del2 * w2' .* Z1 .* (1-Z1);
        del3 = del3(:, 2:size(del3,2));
        % Updating weight
        w1 = w1 - (eta/siz)*X_train'*del3;
    end
    %% Computing the training and validation error
    
    ydash = MLP_Test(X, w1, w2, v, d_Per);
    yydash = MLP_Test(X_val, w1, w2, v, d_Per);
    train_error(epoch) = sum((Y - ydash).^2)/size(Y,1);
    val_error(epoch) = sum((Y_val - yydash).^2)/size(Y_val,1);
    
    % displaying error after each epoch
    disp(sprintf('training error after epoch %d: %f\n',epoch,...
        train_error(epoch)));
    disp(sprintf('validation error after epoch %d: %f\n',epoch,...
        val_error(epoch)));
end
 
end

