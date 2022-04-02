from matplotlib import pyplot as plt
import numpy as np

# recommended color for different digits
color_mapping = {0:'red',1:'green',2:'blue',3:'yellow',4:'magenta',5:'orangered',
                6:'cyan',7:'purple',8:'gold',9:'pink'}

def plot2d(data,label,split='train'):
    # 2d scatter plot of the hidden features
    #data will be a tXh (1000X2)

    for t in range(len(data)):
         plt.scatter(data[t][0], data[t][1], color = color_mapping[label[t]])
    plt.xlabel("Hidden 1")
    plt.ylabel("Hidden 2")
    plt.title(split)
    plt.show()
    pass

def plot3d(data,label,split='train'):
    # 3d scatter plot of the hidden features
    # x = data[:,0]
    # y = data[:,1]
    # z = data[:,2]
    # c = np.strings(label.shape)
    # for t in range(len(label)):
    #     c[t] = color_mapping[label[t]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for t in range(len(data)):
        ax.scatter(data[t][0],data[t][1],data[t][2], color = color_mapping[label[t]])
    ax.set_xlabel("Hidden 1")
    ax.set_ylabel("Hidden 2")
    ax.set_zlabel("Hidden 3")
    plt.show()
    pass
