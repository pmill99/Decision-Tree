import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv") #Taking in data from CSV
#data.head()

from numpy import genfromtxt #Convert data into two numpy arrays
X = genfromtxt('data.csv', delimiter=',',skip_header=1,usecols = (0,1)) #X1 and X2
y = genfromtxt('data.csv', delimiter=',',skip_header=1,usecols = (2)) #Classes

import matplotlib   #visualize the dataset
fig = plt.figure(figsize=(15,15))
plt.scatter(X[:, 1], X[:, 0], c=y, cmap=matplotlib.colors.ListedColormap(['red','blue']))
plt.title('Sample Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

def gini(y):
    # calculate gini given the classes of each instance
    i = y.shape[0] #number of items left to iterate over
    counts = dict(zip(*np.unique(y, return_counts = True))) #counts unique instances of each class in dataset
    impurity = 1 - sum((count/i)**2 for count in counts.values()) #iteratively calculate impurity for both classes for the length of the dataset
    return impurity

def information_gain(l_y, r_y, current_gini):
    # determine gain on a given split
    i, j = l_y.shape[0], r_y.shape[0] #m = number of items left, n= number of items right
    w = i / (i + j) #weighting impurities
    return current_gini - w * gini(l_y) - (1 - w) * gini(r_y)

def get_split(X, y):
    best_gain, best_ind, best_value = 0, None, None
    current_gini = gini(y)
    num_features = X.shape[1] #tracks how many features are being tested for split 

    for ind in range(num_features):  

        instances = np.unique(X[:, ind], return_counts = False)  #collect feature values for testing for best split

        for value in instances:  

            left, right = test_split(ind, value, X, y)

            if right['y'].shape[0] == 0 or left['y'].shape[0] == 0:
                continue

            gain = information_gain(left['y'], right['y'], current_gini)

            if gain > best_gain:
                best_gain, best_ind, best_value = gain, ind, value
    best_split = {'gain': best_gain, 'ind': best_ind, 'value': best_value}
    return best_split

def test_split(ind, value, X, y):
    # split a group of examples based on given index (feature) and value
    spmask = X[:, ind] < value 
    left = {'X': X[spmask, :], 'y': y[spmask]} #left = X's > a certain value
    right = {'X': X[~spmask, :], 'y': y[~spmask]} #left = X's < a certain value
    return left, right

class Decision_Node:
    # define a decision node
    def __init__(self, ind, value, left, right):
        self.ind, self.value = ind, value
        self.left, self.right = left, right

class Leaf:
    # define a leaf node
    def __init__(self, y):
        self.counts = dict(zip(*np.unique(y, return_counts = True))) # a leaf contains instances of the classes
        self.prediction = max(self.counts.keys(), key = lambda x: self.counts[x]) #prediction of class for each instance

def decision_tree(X, y, max_dep, min_size):
    correct = 0 #Keep track of how many predictions come back successful
    
    def build_tree(X, y, dep, max_dep = max_dep, min_size = min_size):
        split = get_split(X, y)

        if split['gain'] == 0 or dep >= max_dep or y.shape[0] <= min_size:
            nonlocal correct
            leaf = Leaf(y)
            correct += leaf.counts[leaf.prediction]
            return leaf

        left, right = test_split(split['ind'], split['value'], X, y)

        left_node = build_tree(left['X'], left['y'], dep + 1)
        right_node = build_tree(right['X'], right['y'], dep + 1)


        return Decision_Node(split['ind'], split['value'], left_node, right_node)
    
    root = build_tree(X, y, 0)
    return correct/y.shape[0], root

#Compare accuracy of model with infinite nodes to model with limited nodes
o_accuracy, o_model = decision_tree(X, y, float('inf'), 1)
accuracy, model = decision_tree(X, y, 4, 1)


#OUTPUT
#For Tree
def print_tree(node, indent = "|---"):
    if isinstance(node, Leaf):
        print(indent + 'Class:', node.prediction)
        return

    print(indent + 'feature_' + str(node.ind) + 
           ' <= ' + str(round(node.value, 2)))
    print_tree(node.left, '|   ' + indent)

    print(indent + 'feature_' + str(node.ind) + 
           ' > ' + str(round(node.value, 2)))
    print_tree(node.right, '|   ' + indent)

print_tree(model)
print(f'The accuracy of a model with unlimited splits is {o_accuracy*100} %')
print(f'With a 4 node split, the accuracy is {accuracy*100} %')

#For predictions
def predict(x, node): #descend tree of nodes until you reach a leaf and return its prediction
    if isinstance(node, Leaf):
        return node.prediction
    
    if x[node.ind] < node.value:
        return predict(x, node.left)
    else:
        return predict(x, node.right)

u = np.linspace(min(X[:, 0]),max(X[:, 0]), 400)  #Arrange features for output
v = np.linspace(min(X[:, 1]),max(X[:, 1]), 400)

models = [o_model, model]
titles = ['Overfit DB', 'DB with limited split']

fig, axs = plt.subplots(ncols = 2, figsize = (12, 5))
for k, ax in enumerate(axs):
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = predict([u[i], v[j]], models[k])    #Fill in predictions

    z = np.transpose(z)

    ax.contourf(u,v,z, alpha = 0.2, levels = 1, antialiased = True)   


    ax.set_title(titles[k])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
plt.show()
