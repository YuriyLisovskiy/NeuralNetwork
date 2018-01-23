# A number of iterations for training

# Lower value increases training speed but causes low performance of network
# Higher value decreases learning speed and increases network performance
EPOCHS = 10000


# A parameter which indicates a speed of network training.

# Lower value causes low training speed but the network will be able to solve
#   more common problems after learning.
# Higher value causes high training speed and but the network will be able
#   to solve only problems from training case.
LEARNING_RATE = 0.007


# A topology of network.

# A size of an array indicates a number of network layers.
LAYERS = [3, 8, 1]
