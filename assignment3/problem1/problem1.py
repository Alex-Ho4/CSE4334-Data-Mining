import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from numpy.linalg import norm


class LRModel(object):

    def __init__(self, learning_rate, epochs):

        self.lr = learning_rate
        self.epochs = epochs
        self.w = np.random.rand(1+2)
        self.allErrors = []
        self.gradNorms = []
        self.iterations = 0        

    def batchTrain(self, X, y):
    
        self.w = np.random.rand(1+2)
        self.allErrors = []
        self.gradNorms = []
        self.iterations = 0        

        for _ in range(self.epochs):
            
            # Set error and gradient to 0
            error = 0
            grad = 0

            for xi, label in zip(X, y):

                # add bias
                xi = np.insert(xi, 0, 1)

                # output prediction of data
                out = self.net(xi)

                # calculate the error gradient
                grad += ((out - label) * xi)

                # add up errors
                error += self.crossEntropy(out, label)

            # Normalize error and error gradient

            error = error / len(X[:, 1])
            grad = grad / len(X[:, 1])

            # calculate the changes of weight (-lr * error gradient)
            cWeight = grad*(-self.lr)

            # update = weight + grad
            for i in range(3):
                self.w[i] += cWeight[i]

            # Add error to list of errors, update which increment it's on
            self.allErrors.append(error)
            self.iterations += 1

            # calculate the l1 norm of the error gradient
            l1 = norm(grad, 1)

            self.gradNorms.append(l1)

            if l1 < 0.001:
                return self

        return self
        
    # X is the training data, y is the class
    def onlineTrain(self, X, y):

        self.w = np.random.rand(1+2)
        self.allErrors = []
        self.gradNorms = []
        self.iterations = 0        

        for _ in range(self.epochs):            

            for xi, label in zip(X, y):

                # add bias
                xi = np.insert(xi, 0, 1)

                # output prediction of data
                out = self.net(xi)

                # calculate the error gradient
                grad = ((out - label) * xi)

                # calculate the changes of weight (-lr * error gradient)
                cWeight = grad*(-self.lr)

                # update = weight + grad
                for i in range(3):
                    self.w[i] += cWeight[i]

                # Add error to list of errors, update which increment it's on
                error = self.crossEntropy(out, label)
                self.allErrors.append(error)
                self.iterations += 1

                # calculate the l1 norm of the error gradient
                l1 = norm(grad, 1)

                self.gradNorms.append(l1)

                if l1 < 0.001:
                    return self

        return self

    # Prediction function
    def prediction(self, xi):
        xi = np.insert(xi, 0, 1)
        out = self.net(xi)

        if out < 0.5:
            return 0
        return 1
        
    # Net function
    def net(self, X):
        return self.sigmoid(np.dot(self.w.transpose(),X))

    # Sigmoid Function (Activation)
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))

    # Cross Entropy Function (Error) [output, y]
    def crossEntropy(self, o, y):
        return -y*np.log(o) - (1-y)*np.log(1-o)    

def main():
    
    # Create training and test data
    mu1 = np.array([1, 0])

    mu2 = np.array([0, 1.5])

    sigma1 = np.array([[1, 0.75],
                        [0.75, 1]])

    sigma2 = np.array([[1, 0.75],
                        [0.75, 1]])

    train0 = np.random.multivariate_normal(mu1, sigma1, 500)
    train1 = np.random.multivariate_normal(mu2, sigma2, 500)

    test0 = np.random.multivariate_normal(mu1, sigma1, 500)
    test1 = np.random.multivariate_normal(mu2, sigma2, 500)

    trainData = np.append(train0, train1, axis = 0)
    trainLabels = np.append(np.zeros(500), np.ones(500), axis = 0)

    testData = np.append(test0, test1, axis = 0)
    testLabels = np.append(np.zeros(500), np.ones(500), axis = 0)
    
    # Problem 1.1: Batch Training
    '''
    learningRates = [1, 0.1, 0.01, 0.001]

    for lr in range(4):

        p1 = LRModel(learningRates[lr], 100000)
        p1.batchTrain(trainData, trainLabels)

        accuracy = 0

        # Plot the predicted classes from test data & calculate accuracy
        class0 = np.empty((1, 2))
        class1 = np.empty((1, 2))

        for i in range(1000):
            
            prediction = p1.prediction(testData[i,:])

            # Fill class arrays
            if prediction == 0:
                class0 = np.append(class0, [testData[i,:]], axis = 0)
            else:
                class1 = np.append(class1, [testData[i,:]], axis = 0)

            # Calculate accuracy
            if prediction == testLabels[i]:
                accuracy += 1

        class0 = np.delete(class0, 0, axis = 0)
        class1 = np.delete(class1, 0, axis = 0)

        accuracy = accuracy / 1000

        # Print Stats

        print("BATCH: Learning rate: %.3f, Accuracy: %f, Iterations: %d" % (learningRates[lr], accuracy, p1.iterations))

        plt.plot(class0[:, 0], class0[:, 1], 'bo')
        plt.plot(class1[:, 0], class1[:, 1], 'ro')

        # Draw Decision Boundary
        
        ax = plt.gca()
        ax.autoscale(False)
        xVals = np.array(ax.get_xlim())
        yVals = -(xVals * p1.w[1] + p1.w[0]) / p1.w[2]
        
        plt.plot(xVals, yVals, 'k--')
        plt.suptitle("BATCH: Learning Rate = %.3f Iterations = %d; Accuracy = %.4f" % (learningRates[lr], p1.iterations, accuracy))
        plt.title("Test Data + Decision Boundary", fontsize=16)

        plt.show()
        plt.close()

        # Plot training error w.r.t. iteration

        xVals = p1.allErrors[:]
        yVals = range(p1.iterations)

        plt.plot(yVals, xVals)
        plt.suptitle("BATCH: Learning Rate = %.3f Iterations = %d; Accuracy = %.4f" % (learningRates[lr], p1.iterations, accuracy))
        plt.title("Training Error w.r.t. Iterations", fontsize=16)
        plt.xlabel("Iterations")
        plt.ylabel("Training Error")

        plt.show()
        plt.close()

        # Plot changes of norm of gradient w.r.t. iteration

        xVals = p1.gradNorms[:]
        yVals = range(p1.iterations)

        plt.plot(yVals, xVals)
        plt.suptitle("BATCH: Learning Rate = %.3f Iterations = %d; Accuracy = %.4f" % (learningRates[lr], p1.iterations, accuracy))
        plt.title("Gradient Norms w.r.t. Iterations", fontsize=16)
        plt.xlabel("Iterations")
        plt.ylabel("Gradient Norms")

        plt.show()
        plt.close()
    '''
    # Problem 1.2: Online Training    
    
    learningRates = [1, 0.1, 0.01, 0.001]

    for lr in range(4):

        p2 = LRModel(learningRates[lr], 100000)
        p2.onlineTrain(trainData, trainLabels)

        accuracy = 0

        # Plot the predicted classes from test data & calculate accuracy
        class0 = np.empty((1, 2))
        class1 = np.empty((1, 2))

        for i in range(1000):
            
            prediction = p2.prediction(testData[i,:])

            # Fill class arrays
            if prediction == 0:
                class0 = np.append(class0, [testData[i,:]], axis = 0)
            else:
                class1 = np.append(class1, [testData[i,:]], axis = 0)

            # Calculate accuracy
            if prediction == testLabels[i]:
                accuracy += 1

        class0 = np.delete(class0, 0, axis = 0)
        class1 = np.delete(class1, 0, axis = 0)

        accuracy = accuracy / 1000

        # Print Stats

        print("ONLINE: Learning rate: %.3f, Accuracy: %f, Iterations: %d" % (learningRates[lr], accuracy, p2.iterations))

        plt.plot(class0[:, 0], class0[:, 1], 'bo')
        plt.plot(class1[:, 0], class1[:, 1], 'ro')

        # Draw Decision Boundary
        
        ax = plt.gca()
        ax.autoscale(False)
        xVals = np.array(ax.get_xlim())
        yVals = -(xVals * p2.w[1] + p2.w[0]) / p2.w[2]
        
        plt.plot(xVals, yVals, 'k--')

        plt.suptitle("ONLINE: Learning Rate = %.3f Iterations = %d; Accuracy = %.4f" % (learningRates[lr], p2.iterations, accuracy))
        plt.title("Test Data + Decision Boundary", fontsize=16)
 
        plt.show()
        plt.close()

        # Plot training error w.r.t. iteration

        xVals = p2.allErrors[:]
        yVals = range(p2.iterations)

        plt.plot(yVals, xVals)
        plt.suptitle("ONLINE: Learning Rate = %.3f Iterations = %d; Accuracy = %.4f" % (learningRates[lr], p2.iterations, accuracy))
        plt.title("Training Error w.r.t. Iterations", fontsize=16)
        plt.xlabel("Iterations")
        plt.ylabel("Training Error")

        plt.show()
        plt.close()

        # Plot changes of norm of gradient w.r.t. iteration

        xVals = p2.gradNorms[:]
        yVals = range(p2.iterations)

        plt.plot(yVals, xVals)
        plt.suptitle("ONLINE: Learning Rate = %.3f Iterations = %d; Accuracy = %.4f" % (learningRates[lr], p2.iterations, accuracy))
        plt.title("Gradient Norms w.r.t. Iterations", fontsize=16)
        plt.xlabel("Iterations")
        plt.ylabel("Gradient Norms")

        plt.show()
        plt.close()

if ( __name__ == '__main__' ):
    main()