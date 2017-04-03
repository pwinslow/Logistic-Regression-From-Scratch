# Import division from future
from __future__ import division

# Analysis imports
import numpy as np
import scipy.optimize


class LogReg(object):
	
	"""

This class implements a logistic regression classifier. 


	"""
	
	__version__ = 'Beta_1.0'
	

	def __init__(self, lmbda = None, threshold = None):

		self.lmbda = lmbda or 0
		self.threshold = threshold or 0.5
		
		'''

		Attributes:
		----------

			lmbda: Ridge regression parameter for penalised optimization of least squares functions.
				 Penalisation shrinks values of regression coefficients during optimization of least squares function.

			threshold: Probability threshold above which a set of inputs is considered to belong to a given class
			
		
		Methods:
		-------
		
			sigmoid: Classic sigmoid function
			
			cost_function: Takes regression parameters, design matrix, and target vector as input and outputs cost function and its gradient.
			
			fit: Takes initial regression parameters, design matrix, and target vector as input. Uses fmin_ncg from scipy.optimize to determine
				and return learned regression parameters.
			
			predict_prob: Takes learned regression parameters and design matrix as input and returns the predicted class probability.
			
			predict: Takes learned regression parameters and design matrix as input and returns predicted class based on whether class probability 
				    is above/below the threshold attribute.
			
			score: Takes learned regression parameters, design matrix, and target vector as input. Calculates and returns accuracy score for the 
				  model based on the percentage of predictions that are correct.
						
			poly_map: Takes in two feature vectors and a desired polynomial degree as input. Creates polynomial feature vectors based on the two
					input feature vectors, concatenates them all together with the input feature vectors to form a non-linear design matrix, 
					adds a bias column to this matrix, and returns the result.
			
					
		'''



	# Define sigmoid function
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	# Define method to calculate both cost function and its gradient
	def cost_function(self, theta, x, y):

		# Initialize necessary values
		m = y.shape[0]
		J = 0
		grad = np.zeros(theta.shape)

		# Calculate regularized cost function (regularization will be specified as an option upon runtime)
		J = np.sum( - np.dot( y.T, np.log( self.sigmoid( np.dot(x, theta) ) ) ) \
			- np.dot( (1-y).T, np.log( 1 - self.sigmoid( np.dot(x, theta) ) ) ) ) / m \
			+ (self.lmbda / 2 / m) * ( np.sum(theta**2) - theta[0]**2 )
			
		# Calculate regularized grad function
		grad = np.dot( x.T, self.sigmoid( np.dot( x, theta) ) - y ) / m
		grad[1:] += (self.lmbda / m) * theta[1:]

		# Return both
		return [J, grad]

	# Define the fit method
	def fit(self, ini_theta, X, Y):
		
		# Define method to return cost based on theta alone
		def theta_cost(theta):
			return self.cost_function(theta, X, Y)[0]

		# Define method to return gradient based on theta alone
		def theta_grad(theta):
			return self.cost_function(theta, X, Y)[1]

		# Determine optimal theta parameters
		optimal_theta = scipy.optimize.fmin_ncg(theta_cost, ini_theta, fprime = theta_grad, maxiter = 400, disp = False)

		# Return the optimal model parameters
		return optimal_theta


	# Define a method to predict whether a label is 0 or 1 using the learned regression parameters
	def predict_prob(self, theta, x):

		# Return actual probability 
		return self.sigmoid( np.dot(x, theta) )
		
		
	# Define a method to predict whether a label is 0 or 1 using the learned regression parameters
	def predict(self, theta, x):
		
		# If sigmoid(X*theta) >= threshold then predict 1, otherwise predict 0
		return 0.5 * ( 1 + np.sign( self.predict_prob(theta, x) - self.threshold ) )


	# Define a method to score the model
	def score(self, theta, x, y):

		return np.mean( self.predict(theta, x) == y )
		
	
	# Define a method for mapping a dataset with only linear features to one with polynomial feature representation
	def poly_map(self, x1, x2, deg):
		
		# Define an array of polynomial features using nested list comprehensions
		poly_x = np.array([x1**(i-j) * x2**j 
                    for i in range(1,int(deg)+1) for j in range(i+1)
                   ])
		
		# Insert the bias column
		poly_x = np.insert(poly_x, 0, 1, axis = 0).T
		
		return poly_x