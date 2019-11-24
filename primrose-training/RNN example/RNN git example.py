import copy, numpy as np

np.random.seed(0)


# compute sigmoid nonlinearity
def sigmoid(x):
    output=1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_derv(output):
    return output * (1 - output)

class Binary_Data: # generates binary sum examples
    def __init__(self,bit_size):
        self.binary_dict = {}
        binary_dim=  bit_size
        self.largest_number=pow(2, binary_dim) #maximal number represented by bit size
        self.binary=np.unpackbits(
            np.array([range(self.largest_number)], dtype=np.uint8).T, axis=1)
        for i in range(self.largest_number):
            self.binary_dict[i]=self.binary[i]
    def int2binary(self,integer):
        return self.binary_dict[integer]
    def generate_data(self):
        self.first_int = np.random.randint(self.largest_number/2)
        self.second_int = np.random.randint(self.largest_number/2)
        self.sum_int = self.first_int + self.second_int
        self.first_bin=self.binary_dict[self.first_int]
        self.second_bin=self.binary_dict[self.second_int]
        self.sum_bin=self.binary_dict[self.sum_int]

# training dataset generation
binary_dim = 8
binary_gen = Binary_Data(binary_dim)
binary_gen.generate_data()

# input variables
alpha=0.1
input_dim=2
hidden_dim=8
output_dim=1

# initialize neural network weights
w_0=2 * np.random.random((input_dim, hidden_dim)) - 1 # input layer weights
w_2=2 * np.random.random((hidden_dim, output_dim)) - 1 # output layer weights
w_h=2 * np.random.random((hidden_dim, hidden_dim)) - 1 # hidden layers weights

# weight update variables
w_0_update=np.zeros_like(w_0)
w_2_update=np.zeros_like(w_2)
w_h_update = np.zeros_like(w_h)

# Train Network
for j in range(1000):
    # generate a simple addition problem (a + b = c)
    a_int=binary_gen.first_int
    a= binary_gen.first_bin # binary encoding

    b_int = binary_gen.second_int
    b=binary_gen.second_bin  # binary encoding

    # answer
    c_int=a_int + b_int
    c=binary_gen.sum_bin

    # store guess (binary encoded)
    d=np.zeros_like(c)

    overallError=0

    layer_2_deltas = []
    layer_h_values = [np.zeros(hidden_dim)]

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        X=np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y=np.array([[c[binary_dim - position - 1]]])
    # forward prop
        # hidden layer (input ~+ prev_hidden)
        layer_h=sigmoid((X @ w_0) + (layer_h_values[-1] @ w_h))

        # output layer (new binary representation)
        layer_2=sigmoid(layer_h @ w_2)

        # calculate error
        layer_2_error=y - layer_2
        layer_2_deltas.append((layer_2_error) * sigmoid_derv(layer_2))
        overallError+=np.abs(layer_2_error[0])

        # decode estimate so we can print it out
        d[binary_dim - position - 1]=np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layer_h_values.append(copy.deepcopy(layer_h))

    future_layer_h_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X=np.array([[a[position], b[position]]])
        layer_h=layer_h_values[-position - 1]
        prev_layer_h=layer_h_values[-position - 2]

        # error at output layer
        layer_2_delta=layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_h_delta=(future_layer_h_delta@(w_h.T) + 
                       layer_2_delta@(w_2.T)) * sigmoid_derv(layer_h)

        # update weights 
        w_2_update += np.atleast_2d(layer_h).T@(layer_2_delta)
        w_h_update += np.atleast_2d(prev_layer_h).T@(layer_h_delta)
        w_0_update += X.T@(layer_h_delta)

        future_layer_h_delta=layer_h_delta

    w_0 += w_0_update * alpha
    w_2 += w_2_update * alpha
    w_h += w_h_update * alpha

    w_0_update*=0
    w_2_update*=0
    w_h_update*=0

    # print out progress
    if (j % 100 == 0):
        print (
        "Error:" + str(overallError))
        print(
        "Pred:" + str(d))
        print(
        "True:" + str(c))
        out=0
        for index, x in enumerate(reversed(d)):
            out+=x * pow(2, index)
        print(
        str(a_int) + " + " + str(b_int) + " = " + str(out))
        print(
        "------------")

