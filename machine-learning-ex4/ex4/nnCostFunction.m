function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    
% Vectorized solution
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

a_1 = [ones(m, 1), X];      % a1 5000x401

z_2 = Theta1 * a_1';
a_2 = sigmoid(z_2); % a2 25x5000
cols = size(a_2, 2);
a_2 = [ones(1, cols); a_2];   % Add a0^(2) 26x5000

a_3 = sigmoid(Theta2 * a_2);  % a3 10x5000

J = trace(-log(a_3) * y_matrix - log(1-a_3) * (1-y_matrix));
J = (J + lambda * (sumsqr(Theta1(:,2:end)) + sumsqr(Theta2(:,2:end))) / 2) ...
    / m;



% % Original solution
% a_1 = [ones(m, 1), X];      % a1 5000x401
% 
% a_2 = sigmoid(Theta1 * a_1'); % a2 25x5000
% cols = size(a_2, 2);
% a_2 = [ones(1, cols); a_2];   % Add a0^(2) 26x5000
% 
% a_3 = sigmoid(Theta2 * a_2);  % a3 10x5000
% 
% y_k = zeros(m, num_labels); % Recode y, 5000x10
% for i = 1:m
%     y_k(i, y(i)) = 1;
% end
% 
% for i = 1:m
%     J = J + -y_k(i,:) * log(a_3(:,i)) - (1-y_k(i,:)) * log(1-a_3(:,i));
% end
% 
% J = J / m;
% 
% % Regularization
% J = J + lambda / (2*m) * (sumsqr(Theta1(:,2:end)) + sumsqr(Theta2(:,2:end)));



% Grad
% Vectorized solution
% Copy forward pass for reference
% a_1 = [ones(m, 1), X];      % a1 5000x401
% 
% a_2 = sigmoid(Theta1 * a_1'); % a2 25x5000
% cols = size(a_2, 2);
% a_2 = [ones(1, cols); a_2];   % Add a0^(2) 26x5000
% 
% a_3 = sigmoid(Theta2 * a_2);  % a3 10x5000

delta_3 = a_3 - y_matrix';
delta_2 = Theta2' * delta_3;

sigmoidGradient_z2 = a_2 .* (1 - a_2);
delta_2 = delta_2 .* sigmoidGradient_z2;
delta_2 = delta_2(2:end,:);

Delta_1 = delta_2 * a_1;
Delta_2 = delta_3 * a_2';

% Original solution
% Delta_1 = 0;
% Delta_2 = 0;
% for t = 1:m
%     % Step 1
%     z_1 = X(t,:)'; % Add +1 unit
%     a_1 = [1; sigmoid(z_1)];
%     
%     z_2 = Theta1 * a_1; % Add +1 unit
%     a_2 = [1; sigmoid(z_2)];
% 
%     z_3 = Theta2 * a_2;
%     a_3 = sigmoid(z_3);
%     
%     % Step 2
%     y_kv = zeros(num_labels, 1);
%     y_kv(y(t)) = 1;
%     delta_3 = a_3 - y_kv;
%     
%     % Step 3
%     delta_2 = Theta2' * delta_3;
%     delta_2 = delta_2(2:end); % Remove delta on +1
%     delta_2 = delta_2 .* sigmoidGradient(z_2);
%     
%     Delta_1 = Delta_1 + delta_2 * a_1';
%     Delta_2 = Delta_2 + delta_3 * a_2';
% end
    
Theta1_grad = Delta_1 / m;
Theta2_grad = Delta_2 / m;

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad + Theta1 * lambda / m;
Theta2_grad = Theta2_grad + Theta2 * lambda / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
