import numpy as np # Used to read in data and for array and matrix operations.
import random # Used to randomly shuffle data points

### Function to Grab Data From Files ###
def get_data():
    """load a CSV file into a useful format for computation"""

    # Loads in Y (numeric label for a picture)
    data = np.genfromtxt('Labels.csv', delimiter=',', skip_header=True)
    Y = data.astype(int)

    # Loads in X (float values representing pixel data)
    data = np.genfromtxt('Values.csv', delimiter=',', skip_header=True)
    X = data.astype(float)

    # Shuffles X and Y pairs
    c = list(zip(X, Y))
    random.shuffle(c)
    X, Y = zip(*c)

    # Splits data into train and test groups
    # 20% Testing, 80% Training
    x_train = X[0:4000]
    x_test = X[4000:]
    y_train = np.array(Y[0:4000])
    y_test = np.array(Y[4000:])
    
    return x_train, y_train, x_test, y_test



### Helper Functions To Be Used in Model ###
def soft_max(x):
    '''
    Applies softmax to a numeric array

    Softmax is a function which converts a numeric vector 
    to a vector of realitive probabilities 
    '''
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def log_loss(weights, X, Y, classes):
    '''
    Returns average log loss of our model to be used while training
    '''
    sum = 0
    for i in range(len(Y)):
        # Calculates probs that x belongs to each class
        prob = soft_max(weights @ X[i])
        for j in range(classes):
            # Updates Loss when y == j 
            if Y[i] == j:
                sum += np.log(prob[j])
    return -sum/len(Y)        



### Model Object ###
class Multivariate_Logistic_Regression:
    '''
    Multivariate Logistic Regression Model that uses
    stochastic gradient descent to update model weights
    '''

    def __init__(self, conv_threshold, n_predictors, n_classes):
        '''
        Initializes Multivariate Logistic Regression Model

        conv_threshold: Determines at what point model training stops
        weights: The weights of the Logistic Regression model
        n_predictors: Number of features Model uses to Predict Outcomes
        n_classes: Number of Buckets to sort Examples into
        '''
        # Preset at initialization
        self.conv_threshold = conv_threshold
        self.n_classes = n_classes
        self.n_predictors = n_predictors
        
        # Preset weights to be a matrix of zeros
        # n possible outcomes, m predictors + 1 bias term
        self.weights = np.zeros((n_classes, n_predictors + 1))


    def train(self, X, Y):
        '''
        Trains our model using stochastic gradient descent, 
        X is a 2D array where each row is an example with n predictors and 
        Y a 1D array which contains labels for the corresponding X example.
        '''
        # Adds bias term to X
        X = np.append(X, np.ones((len(X), 1)), axis=1)

        converge = False

        while not converge:

            # Shuffle training examples
            c = list(zip(X, Y))
            random.shuffle(c)
            X, Y = zip(*c)
            
            # the previous loss needs to be calculated on X and Y
            previous_loss = log_loss(self.weights, X, Y, self.n_classes)
            
            for i in range(len(X)//5):
                # Creates batches with size of 5 for training
                X_batch = X[i*5:(i+1)*5]
                Y_batch = Y[i*5:(i+1)*5]

                # Creates gradient which will be used to update model weights
                L_gradient = np.zeros(self.weights.shape)

                for (x, y) in zip(X_batch, Y_batch):
                    for j in range(self.n_classes):
                        if y == j: 
                            L_gradient[j] += (soft_max(self.weights @ x)[j] - 1) * x
                        else:
                            L_gradient[j] += soft_max(self.weights @ x)[j] * x
                            
                self.weights = self.weights - (.03*L_gradient)/len(X_batch) # Update Model Weights
                
            # the current loss need to be calculated on X and Y
            current_loss = log_loss(self.weights, X, Y, self.n_classes)
            
            # Check if converged
            if abs(current_loss - previous_loss) < self.conv_threshold:
                converge = True
        return


    def predict(self, X):
        '''
        Creates predictions based on trained model weights and predictors in X
        Returns prediction and confidence probabilty of said prediction
        '''
        # Adds bias term to X if X doesn't already have a bias term
        if len(X[0]) != len(self.weights[0]):
            X = np.append(X, np.ones((len(X), 1)), axis=1)

        # Presets pred to be the length of examples of x
        pred = np.zeros(len(X))
        probs = np.zeros(len(X))

        for i in range(len(X)):
            # For each example creates predictions based on class with highest probability
            probabilities = soft_max(self.weights @ X[i])
            pred[i] = np.argmax(probabilities) 
            probs[i] = int(max(probabilities)*100)
        return pred, probs


    def accuracy(self, X, Y):
        '''
        Returns the accuracy of how our model performed on set of given X's with lables Y
        '''
        # Adds bias term to X
        X = np.append(X, np.ones((len(X), 1)), axis=1)

        # Creates predictions
        pred, probs = self.predict(X)
        # Compares predictions to actual values
        acc = sum(pred == Y)/len(Y)
        return acc