import numpy as np
#import time #used only for testing

def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    if mean is not None:
        # directly use the mean and std precomputed from the training data

        return np.divide(np.subtract(data, mean), std+1e-15)
    else:
        # compute the mean and std based on the training data
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)

        return np.divide(np.subtract(data, mean), std+1e-15), mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(len(label)):
        one_hot[i,label[i]] = 1
    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    # You may receive some warning messages from Numpy. No worries, they should not affect your final results
    #works (passes):
    f_x = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    #works:
    #f_x = np.tanh(x)
    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    f_x = np.zeros((len(x), len(x[0])))
    for i in range(len(x)):
        sgma = np.sum(np.exp(x[i]))
        f_x[i] = np.exp(x[i])/sgma
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])
        self.h = num_hid
        self.i = 10
        self.j = 64

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        #truecount = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
        #while count <= 5: #for testing
            #start_time = time.time()
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)

            softy, z = self.predict_guts(train_x)
            #sofy shape = 1000X10
            #z shape = 1000X4

            #blanks
            deltav = np.zeros((self.weight_2.shape))
            deltav0 = np.zeros((self.bias_2.shape))
            deltaw = np.zeros((self.weight_1.shape))
            deltaw0 = np.zeros((self.bias_1.shape))


            #Ranges:
            T = len(train_x) #nX10
            I = self.i
            H = self.h
            J = self.j

            error = np.subtract(train_y, softy)
            #size: 1000X10
            dtanh = np.subtract(1,np.power(z, 2))
            #size 1000X4

            #Commented out update formulas apear to work but are too slow.
            #Second attempt decreased the times indexing the blank tables directly
            #Thrid attempt decreased number of loops
            #Does not time out, still takes forever.

            #Looping backpropagation formulas:
            #Delta V(works)
            # for i in range(I):
            #     for h in range(H):
            #         count = 0
            #         for t in range(T):
            #             count += (train_y[t][i]-softy[t][i]) * z[t][h]
            #         #print("deltaV, H: ", h)
            #         #print("Count : ", lr*count)
            #         #print("inserted at: ", h, " ", i)
            #         deltav[h][i] = lr * count

            for h in range(H): #works
                count = 0
                for t in range(T):
                    deltav[h] += (error[t]) * z[t][h]
            deltav = np.multiply(lr, deltav)

            # #DeltaV0(works)
            # for i in range(I):
            #     #print("Delta V0, I: ", i)
            #     count = 0
            #     for t in range(T):
            #         count += (train_y[t][i]-softy[t][i])
            #     deltav0[0][i] = lr * count

            deltav0 = lr * np.sum(error, axis = 0) #works


            #DeltaW (works)# ~4seconds:
            # for h in range(H):
            #     for j in range(J):
            #         count = 0
            #         for t in range(T):
            #             count2 = 0
            #             for i in range(I):
            #                 count2 += (train_y[t][i]-softy[t][i]) * self.weight_2[h][i]
            #             count2 = count2 * (1-(z[t][h]**2)) * train_x[t][j]
            #             count += count2
            #         deltaw[j][h] = lr * count

            #DeltaW New ~3seconds: (still works, might not be correct)
            for h in range(H): #4
                #print("H: ", h)
                for j in range(J): #64
                    #print("J: ", j)
                    for t in range(T): #1000
                        deltaw[j][h] += np.sum(error[t] * self.weight_2[h]) * dtanh[t][h] * train_x[t][j] #a 1X10 row sumed into a scalar
            deltaw = np.multiply(lr, deltaw)

            # for h in range(H): #4 #So slow it is funcitonally broken.
            #     #print("H: ", h)
            #     for j in range(J): #64
            #         #print("J: ", j)
            #         for t in range(T): #1000
            #             for i in range(I):
            #                 deltaw[j][h] += np.sum(error[t] * self.weight_2[h][i]) * dtanh[t][h] * train_x[t][j] #a 1X10 row sumed into a scalar
            # deltaw = np.multiply(lr, deltaw)


            ##DeltaW0 (works)
            for h in range(H):
                count = 0
                for t in range(T):
                    count2 = 0
                    for i in range(I):
                        count2 += (train_y[t][i]-softy[t][i]) * self.weight_2[h][i]
                    count2 += count2 * (1-(z[t][h]**2)) #was =, changed back was broken
                    count += count2
                deltaw0[0][h] = lr * count


            # # #DeltaW0 (Broken!)
            # for h in range(H):
            #     for t in range(T):
            #         deltaw0[0][h] += np.sum(np.multiply(error[t], self.weight_2[h])) * dtanh[t][h]
            # deltaw0 = np.multiply(lr, deltaw0)

            #testing:
            # truecount = truecount + 1
            # print("truecount: ")
            # print(truecount)
            # print("Delta V: ")
            # print(deltav)
            # print("delta V 0: ")
            # print(deltav0)
            # print("Delta W: ")
            # print(deltaw[:10])
            # print("Delta W 0: ")
            # print(deltaw0)
            # print("Total time for loop: ", time.time() - start_time)


            #update the parameters based on sum of gradients for all training samples
            self.weight_1 = self.weight_1 + deltaw
            self.bias_1 = self.bias_1 + deltaw0
            self.weight_2 = self.weight_2 + deltav
            self.bias_2 = self.bias_2 + deltav0


            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # convert class probability to predicted labels
        softy, z = self.predict_guts(x)
        y = softy.argmax(1)
        return y

    def predict_guts(self, x):
        # generate the predicted probability of different classes
        # this is the one that uses softmax
        z = self.get_hidden(x)
        v = np.dot(z, self.weight_2) + self.bias_2
        softy = softmax(v)
        return softy, z


    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        z = np.matmul(x, self.weight_1) + self.bias_1
        dim = z.shape
        #print("z shape: ", dim)
        tanz = np.zeros(dim)

        for i in range(dim[0]):
            for j in range(dim[1]):
                tanz[i][j] = tanh(z[i][j])
        return tanz

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
