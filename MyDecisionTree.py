import numpy as np

'''
In this programming assignment, we are going to implement the decision tree with recursion. The recommended 
implementation order of the functions are:
1. compute_node_entropy: compute node entropy based with the given labels (sum -p*log2(p+1e-15), 
where p is the probability of each label) - DONE
2. compute_split_entropy: given the left and right labels of the split, first compute the entropy of 
left and right labels with (1), and then weighted combine them to get the split entropy - DONE
3. select_features: given the data and label, iterate through all possible features for split, and use (2) to compute 
the entropy. Select the feature index with best(lowest) entropy - DONE
4. generate_tree: given all the data/label and min_entropy, generate the tree with recursion: the structure could be 
like follow: (Stop Criteria; Find the feature of current split (3); recursively call itself again with
left/right data/labels). With this structure, the function will recursively find the feature and also the feature for 
their left/right children, until the stop criteria is reached
5. Decision_tree.predict: given each test data, you can traverse the tree to find its corresponding labels and return
the labels
--------------------
Here are some clarifications:
For all test, we only test the functionality of each function, please report your answers in the pdf files.
To ensure a deterministic result, don't shuffle data.
'''
class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.is_leaf = False # whether or not the current node is a leaf node
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node)
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x, train_y)

    def predict(self, test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x), ]).astype('int') # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            cur_node = self.root
            testrow = test_x[i]
            while cur_node.is_leaf == False:
                if testrow[cur_node.feature] == 0:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            prediction[i] = cur_node.label
        return prediction

    def generate_tree(self, data, label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node
        if node_entropy < self.min_entropy:
            # determine the class label for leaf node
            val, counts = np.unique(label, return_counts=True)
            cur_node.label = val[np.argmax(counts)]
            cur_node.is_leaf = True
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data, label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature
        train_data = np.c_[data, label]
        train_data0 = train_data[train_data[:, selected_feature] == 0, :]
        train_data1 = train_data[train_data[:, selected_feature] == 1, :]
        data0 = train_data0[:,:-1]
        label0 = train_data0[:,-1].astype('int')
        data1 = train_data1[:,:-1]
        label1 = train_data1[:,-1].astype('int')

        #and start the next level of recursion
        cur_node.left_child = self.generate_tree(data0, label0)
        cur_node.right_child = self.generate_tree(data1, label1)

        return cur_node

    def select_feature(self, data, label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        best_entropy = 100
        for i in range(len(data[0])):
            # compute the entropy of splitting based on the selected features
            a = np.vstack((data[:, i], label)).T # Create array of only the select variable and the label, size  = row,2
            a = a[a[:, 0].argsort()] # Sort he array into 0 and 1
            b = np.split(a[:, 1], np.unique(a[:, 0], return_index=True)[1][1:]) # Split into two arrays
            entropy = 101
            if len(b[0])<len(data): # Make sure second array isn't EMPTY (feature is of one variable, like 0)
                entropy = self.compute_split_entropy(b[0], b[1]) # Feed each array in to get post split entropy
            if entropy == 0:
                best_feat = i
                return best_feat
            if entropy < best_entropy:
                best_feat = i
                best_entropy = entropy
            # select the feature with minimum entropy
        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two branches. Left is 0.
        #split_entropy = -1 # placeholder
        rn = len(right_y)
        ln = len(left_y)
        n = rn + ln #total number of items going to node
        return rn/n * self.compute_node_entropy(right_y) + ln/n * self.compute_node_entropy(left_y)

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        #node_entropy = -1 # placeholder
        uniquevals,count = np.unique(label, return_counts=True) #Count will be a vector of # of each label 1-9
        #print("uniquevals, Count: ", uniquevals,count)
        p = count / count.sum() #p is a 9 value  vector
        return sum(-(p * np.log2(p+1e-15)))

