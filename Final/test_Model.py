from Model import *
from pytest import approx

# Tests for get_data function
def test_get_data():
    # Grabs Data and Seperates it between ind. vs dep. variables
    # and training and testing sets
    x_train, y_train, x_test, y_test = get_data()
    
    # Test x_train contains correct number of examples
    assert len(x_train) == 4000

    # Checks that each example contains 400 data points
    for i in range(4000):
        assert x_train[i].shape == (400,)

    # Test x_test contains correct number of examples
    assert len(x_test) == 1000

    # Checks that each example contains 400 data points
    for i in range(1000):
        assert x_test[i].shape == (400,)

    # Test y_train contains correct number of labels
    assert len(y_train) == 4000
    assert y_test.shape == (4000,)
    # Checks all values belong to 0-9
    assert np.unique(y_train) == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Test y_test contains correct number of labels
    assert len(y_train) == 1000
    assert y_test.shape == (1000,)
    # Checks all values belong to 0-9
    assert np.unique(y_test) == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


# Test for soft_max function
def test_soft_max():
    # Base case when one number is given
    assert soft_max(0) == 1.0
    assert soft_max(42) == 1.0

    # Case when multiple numbers are given in a list format
    # All the same
    assert (soft_max([0,0]) == np.array([0.5, 0.5])).all
    # Different values
    assert (soft_max([0,1,2]) == np.array([0.09003057, 0.24472847, 0.66524096])).all

    # Test when given np.array
    assert (soft_max(np.array([0,0])) == np.array([0.5, 0.5])).all
    assert (soft_max(np.array([0,1,2])) == \
        np.array([0.09003057, 0.24472847, 0.66524096])).all

# Test for log_loss helper function
def test_log_loss():
    # Tests on one example
    x_1 = np.array([[0,0,0,1]])
    test_weights = np.zeros((1, 4))  # 1 Class and 3 Predictors + 1 Bias Term
    y_1 = np.array([0])

    # Test Model Loss
    assert log_loss(test_weights, x_1, y_1, 3) == 0

    # Creates Test Data
    x_2 = np.array([[0,4,1], [0,3,1], [5,0,1], [4,1,1], [0,5,1]])
    test_weights = np.zeros((2, 3)) # 2 Class and 2 Predictors + 1 Bias Term
    y_2 = np.array([0,0,1,1,0])

    # Test Model Loss
    assert log_loss(test_weights, x_2, y_2, 3) == approx(0.693, .001)


### Tests for Multivariate Logistic Regression Model ###

# Test for __init__ Method
def test___init__():
    # Creating Model Object
    a = Multivariate_Logistic_Regression(1,400,10)

    # Checking Model Initializations
    assert a.conv_threshold == 1
    assert (a.weights == np.zeros((10, 401))).all


# Test for train Method
def test_train():
    # Simple Train Model Test with 2 predictors and 2 outcomes and 5 examples
    a = Multivariate_Logistic_Regression(1, 2, 2)
    x_bias = np.array([[0,4], [0,3], [5,0], [4,1], [0,5]])
    y = np.array([0,0,1,1,0])
    a.train(x_bias, y)
    assert (a.weights == np.array([[-0.027,  0.033,  0.003], \
        [0.027, -0.033, -0.003]])).all

    # Simple Train Model Test with 2 predictors and 2 outcomes and 5 examples
    b = Multivariate_Logistic_Regression(1, 2, 2)
    x_bias = np.array([[-1,0], [2,2], [7,0], [9,1], [2,1]])
    y = np.array([0,0,1,1,0])
    b.train(x_bias, y)
    assert (b.weights == np.array([[-0.039,  0.006,  0.003], \
       [0.039, -0.006, -0.003]])).all


# Test for predict Method
def test_predict():
    # Trains simple Model with 2 predictors and 2 outcomes and 5 examples
    a = Multivariate_Logistic_Regression(1, 2, 2)
    x_bias = np.array([[-1,0], [2,2], [7,0], [9,1], [2,1]])
    y = np.array([0,0,1,1,0])
    a.train(x_bias,y)
    
    # Tests predict on one example
    assert a.predict([[0,0,1]])[0] == 0
    assert a.predict([[-5,3,1]])[0] == 0
    assert a.predict([[9,0,1]])[0] == 1

    # Tests predict on list of examples
    assert (a.predict([[0,0], [-5,3], [9,0]])[0] == [0,0,1]).all


# Test for accuracy Method
def test_accuracy():
    # Trains simple Model with 2 predictors and 2 outcomes and 5 examples
    a = Multivariate_Logistic_Regression(1, 2, 2)
    x_bias = np.array([[-1,0], [2,2], [7,0], [9,1], [2,1]])
    y = np.array([0,0,1,1,0])
    a.train(x_bias,y)

    # Example with 100% Accuracy
    x_bias_test = [[0,0], [-5,3], [9,0]]
    y_test = [0,0,1]
    assert a.accuracy(x_bias_test, y_test) == 1

    # Example with 75% Accuracy
    x_bias_test = [[0,0], [-5,3], [9,0], [4,3]]
    y_test = [0,0,0,1]
    assert a.accuracy(x_bias_test, y_test) == .75

    # Example with 0% Accuracy
    x_bias_test = [[0,0], [-5,3], [9,0]]
    y_test = [1,1,0]
    assert a.accuracy(x_bias_test, y_test) == 0






