

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


'''
PROMPT:
A logistic regression model with cross-validation is initialized using LogisticRegressionCV from sklearn.linear_model.
The model is then trained on the dataset X and y using the fit method.'''

# Initialize logistic regression model with cross-validation
model = sklearn.linear_model.LogisticRegressionCV()

# Train the model on the dataset
model.fit(X, y)


'''
PROMPT:
The decision boundary is plotted using the plot_decision_boundary function, passing a lambda function that uses the model's predict method.
The plot is titled "Logistic Regression".
The plot is displayed using plt.show().
'''

# Plot the decision boundary
plot_decision_boundary(lambda x: model.predict(x))
plt.title("Logistic Regression")
plt.show()

