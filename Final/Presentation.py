from Model import * # Importing Model
import matplotlib.pyplot as plt # Used for displaying plots 

### Understanding the Data ###

# Set Random seed for deterministic behavior
random.seed(2)
np.random.seed(2)

# Reads data sets and creates training and testing sets
X, Y, XT, YT = get_data()

# Image of one Example
def example_image(n):
    '''
    Produces a label example image of a number and the 
    models prediction for what number it is
    '''
    a = np.reshape(X[n], (20, 20), order='F')
    plt.imshow(a, cmap='bone_r', interpolation='nearest')
    plt.tick_params(labelleft = False, labelbottom = False)
    plt.title (f'Label: {Y[n]}', color = 'white')
    plt.show()

example_image(0)
example_image(1)
example_image(2)


### Running the Model ###

### Creates Multivariate Logistic Regression Obj. ###
threshold = 1e-4
n_preds = len(X[0])  # 400 Predictors : 1 for each pixel
n_outcomes = len(np.unique(Y))   # 10 Outcomes : Numbers 0-9 

model = Multivariate_Logistic_Regression(threshold, n_preds, n_outcomes)

# Trains model
model.train(X, Y)

def weights_image(n):
    '''
    Produces a visualization of model weights for a specified number (n) 
    '''
    a = np.reshape(model.weights[n][0:400], (20, 20), order='F')
    plt.imshow(a, cmap='Greys', interpolation='nearest')
    plt.tick_params(labelleft = False, labelbottom = False)
    plt.title (f'Model Weights for {n}' , color = 'white')
    plt.show()

# Visualize Model Weights
weights_image(0)
weights_image(1)
weights_image(2)

# Reports Accuracy and Predictions and Probabilty of Confidence for Each Prediction
predictions, probs = model.predict(XT) # Creates Predictions using Test Examples 
acc = model.accuracy(XT, YT) * 100
print(f'Model has an Accuracy of {acc}%')


### Data Visualizations ###

# Creates Plot to Visualize Error for Each Number
labs = []
diff = []
# Fills in labels and differences with Model Values
for i in range(10):
    labs.append(str(i))
    diff.append(sum(predictions == i) - sum(YT == i))
# Plot Model Error for Each Number
plt.title ('Model Error for Each Number')
plt.xlabel("Number")
plt.ylabel('Actual Count - Predicted Count')
plt.bar(labs, diff)
plt.show()


def prediction_image(n):
    '''
    Produces a label example image of a number and the 
    models prediction for what number it is
    '''
    a = np.reshape(XT[n], (20, 20), order='F')
    plt.imshow(a, cmap='bone_r', interpolation='nearest')
    plt.tick_params(labelleft = False, labelbottom = False)
    plt.title (f'Actual: {YT[n]}   Prediction: {predictions[n]}   Conf: {probs[n]}%', color = 'white')
    plt.show()

# Creates 5 Examples of Predictions on a given image
prediction_image(0)
prediction_image(1)
prediction_image(2)
prediction_image(3)
prediction_image(11)
