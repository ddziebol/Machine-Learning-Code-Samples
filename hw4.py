#import libraries
import numpy as np
from MyDecisionTree import Decision_tree
import matplotlib.pyplot as plt

# read in data.
# training data
train_data = np.genfromtxt("optdigits_train.txt",delimiter=",")
train_x = train_data[:,:-1]
train_y = train_data[:,-1].astype('int')

# validation data
valid_data = np.genfromtxt("optdigits_valid.txt",delimiter=",")
valid_x = valid_data[:,:-1]
valid_y = valid_data[:,-1].astype('int')

# test data
test_data = np.genfromtxt("optdigits_test.txt",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1].astype('int')

############### Problem a ###################
# experiment with different settings of minimum node entropy
candidate_min_entropy = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1, 2.0]
valid_accuracy = []
trainE = np.array([0,0])
validE = np.array([0,0])
for i, min_entropy in enumerate(candidate_min_entropy):
    # initialize the model
    clf = Decision_tree(min_entropy=min_entropy)
    # update the model based on training data, and record the best validation accuracy
    clf.fit(train_x,train_y)
    predictions_train = clf.predict(train_x)
    predictions_val = clf.predict(valid_x)
    cur_train_accuracy = np.count_nonzero(predictions_train.reshape(-1)==train_y.reshape(-1))/len(train_x)
    cur_valid_accuracy = np.count_nonzero(predictions_val.reshape(-1)==valid_y.reshape(-1))/len(valid_x)
    valid_accuracy.append(cur_valid_accuracy)
    trainE = np.vstack([trainE, [min_entropy, cur_train_accuracy]])
    validE = np.vstack([validE, [min_entropy, cur_valid_accuracy]])
    print('Training/validation accuracy for minimum node entropy %f is %.3f / %.3f' %(candidate_min_entropy[i],cur_train_accuracy,cur_valid_accuracy))

# select the best minimum node entropy and use it to train the model
best_entropy = candidate_min_entropy[np.argmax(valid_accuracy)]
clf = Decision_tree(min_entropy=best_entropy)
clf.fit(train_x,train_y)

# evaluate on test data
predictions = clf.predict(test_x)
accuracy = np.count_nonzero(predictions.reshape(-1)==test_y.reshape(-1))/len(test_x)

print('Test accuracy with minimum node entropy %f is %.3f' %(best_entropy,accuracy))

np.savetxt("train_entropy.csv", trainE, delimiter=",", fmt="%.3f", header = "Theta, T Entropy", comments="")
np.savetxt("validation_entropy.csv", validE, delimiter=",", fmt="%.3f", header = "Theta, E Entropy", comments="")

plt.title("Testing and Validation data")
plt.xlabel("Theta")
plt.ylabel("Accuracy")
plt.plot(trainE[1:,0], trainE[1:,1], label = "Training accuracy", color = "green")
plt.plot(validE[1:,0], validE[1:,1], label = "Validation accuracy", color = 'red')
plt.legend()
plt.show()