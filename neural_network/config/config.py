# A number of iterations for training

# Lower value increases training speed but causes low performance of network
# Higher value decreases learning speed and increases network performance
ITERATIONS = 10000


# A parameter which indicates a speed of network training.

# Lower value causes low training speed but the network will be able to solve
#   more common problems after learning.
# Higher value causes high training speed and the network will be able
#   to solve only problems from training case.
LEARNING_RATE = 0.007


# A topology of network.

# The first member of an array is a number of input connections,
INPUT_LAYER = [3]

# An amount of hidden layers.
# Every next one should be less or equal to 2 to the power of previous layer connections, i.e.,
# input layer is [3], then first hidden layer must be <= 2^3, second layer must be <= 2^(2^3), etc.
HIDDEN_LAYERS = [8, 5]

# The output has only one connection.
OUTPUT_LAYER = [1]
