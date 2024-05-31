
#REF=https://dennybritz.com/posts/wildml/implementing-a-neural-network-from-scratch/

'''
PROMPT:
The numpy library is imported as np.
The sklearn library is imported along with specific modules.
The matplotlib library and its pyplot module are imported for plotting.
sklearn.datasets and sklearn.linear_model are imported for dataset generation and linear modeling, respectively.
'''


import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.linear_model

# PROMPT:
# Write a Python function named `calculate_loss` that computes the loss for a given model.
# The function should take a dictionary `model` as input, which contains the parameters 'W1', 'b1', 'W2', and 'b2'.
# It should also use global variables `X`, `y`, `num_examples`, and `reg_lambda`.
# The loss is computed as follows:
# 1. Perform a forward pass through the neural network:
#    - Compute the first layer activations `z1` by multiplying `X` with `W1` and adding `b1`.
#    - Apply the `tanh` activation function to `z1` to get `a1`.
#    - Compute the second layer activations `z2` by multiplying `a1` with `W2` and adding `b2`.
# 2. Calculate the probabilities `probs` using the softmax function.
# 3. Compute the data loss using the negative log likelihood of the probabilities of the correct class labels.
# 4. Add regularization to the data loss using `reg_lambda` and the L2 norms of `W1` and `W2`.
# 5. Return the average loss over all examples.

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# PROMPT:
# Write a Python function named `predict` that makes predictions using a given model.
# The function should take a dictionary `model` and a numpy array `x` as inputs.
# The `model` dictionary contains the parameters 'W1', 'b1', 'W2', and 'b2'.
# The function should perform the following steps:
# 1. Compute the first layer activations `z1` by multiplying `x` with `W1` and adding `b1`.
# 2. Apply the `tanh` activation function to `z1` to get `a1`.
# 3. Compute the second layer activations `z2` by multiplying `a1` with `W2` and adding `b2`.
# 4. Calculate the probabilities `probs` using the softmax function.
# 5. Return the predicted class labels as the indices of the maximum probability for each example.
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# PROMPT:
# Write a Python function named `build_model` that trains a neural network model.
# The function should take the following parameters:
# - `nn_hdim`: the number of neurons in the hidden layer
# - `num_passes`: the number of training iterations (default is 20000)
# - `print_loss`: a boolean indicating whether to print the loss during training (default is False)
# The function should use global variables `X`, `y`, `num_examples`, `nn_input_dim`, `nn_output_dim`, `reg_lambda`, and `epsilon`.
# The training process involves the following steps:
# 1. Initialize the model parameters `W1`, `b1`, `W2`, and `b2` with random values.
# 2. For each training iteration:
#    - Perform a forward pass through the network to compute the layer activations and output probabilities.
#    - Compute the gradients using backpropagation.
#    - Update the model parameters using gradient descent.
# 3. If `print_loss` is True, print the loss every 1000 iterations using the `calculate_loss` function.
# 4. Return the trained model as a dictionary containing `W1`, `b1`, `W2`, and `b2`.
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}
    
    for i in range(0, num_passes):

        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        if print_loss and i % 1000 == 0:
          print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))
    
    return model

'''
PROMPT:
The function plot_decision_boundary takes a prediction function pred_func as input.
It sets the minimum and maximum values for the grid with some padding.
It generates a grid of points with a specified distance h between them.
It predicts the function values for the entire grid using pred_func.
It reshapes the predicted values to match the grid shape.
It plots the decision boundary using plt.contourf and the training examples using plt.scatter.
'''

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


'''
PROMPT:
The figure size is set to 10x8 inches.
A dataset of 200 points is generated using make_moons with a noise level of 0.20.
The data points are plotted using plt.scatter, with colors corresponding to the classes in y. The colormap used is plt.cm.Spectral.
'''

# Set the figure size
plt.figure(figsize=(10, 8))

# Generate the dataset
X, y = sklearn.datasets.make_moons(n_samples=200, noise=0.20)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Display the plot
#plt.show()

#PROMPT: Define the number of examples in the training set
num_examples = len(X)

#PROMPT: Set the dimensionality of the input and output layers
nn_input_dim = 2
nn_output_dim = 2

#PROMPT: Configure the parameters for gradient descent
epsilon = 0.01 
reg_lambda = 0.01 


#PROMPT:
# Use the `build_model` function to train a neural network model with 3 neurons in the hidden layer.
# Set `print_loss` to True to print the loss during training.
model = build_model(3, print_loss=True)


#PROMPT:
# Plot the decision boundary using the `plot_decision_boundary` function.
# The decision boundary should be based on the predictions made by the `predict` function using the trained `model`.
# Set the title of the plot to "Decision Boundary for hidden layer size 3".
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
#plt.show()


#PROMPT:
# Create a plot to visualize decision boundaries for neural networks with different hidden layer sizes.
# Set the figure size to (16, 32).
# Create a list of different hidden layer sizes to be used in building neural network models.
# For each hidden layer size in the list `hidden_layer_dimensions`, do the following:
# 1. Create a subplot for the current hidden layer size.
# 2. Set the title of the subplot to 'Hidden Layer size %d' with the current hidden layer size.
# 3. Build a model using the `build_model` function with the current hidden layer size.
# 4. Plot the decision boundary using the `plot_decision_boundary` function and the `predict` function with the trained model.
# After looping through all hidden layer sizes, display the plot.
plt.figure(figsize=(16, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model = build_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()