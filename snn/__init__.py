import numpy as np

class perceptron:
    """Simple class for creating basic NN"""

    def __init__(self, *layers):
        """Constructor. It takes tuple which contains info about number of
        neurons in each layer"""

        #checking arguments
        if len(layers) < 3:
            raise ValueError('Could not create perceptron with less than 3 layers')

        for val in layers:
            if type(val) != int:
                raise TypeError('Not an integer number')
            elif val < 1:
                raise ValueError('Invalid number of neurons in layer')

        #creating synapses and biases
        self.__syn__ = list()
        self.__biases__ = list()

        for i in range(len(layers) - 1):
            syn_size = (layers[i + 1], layers[i])
            new_syn = 2 * np.random.random(syn_size) - 1
            self.__syn__.append(new_syn)
            self.__biases__.append(0)

        #create variable for last training
        self.__layers__ = list()

        #set learning rate
        self.__rate__ = 0.01

    def lrate(self):
        """Getter for learning rate"""
        return self.__rate__

    def set_lrate(self, new_rate):
        """Setter for learning rate"""
        if type(new_rate) != float:
            raise TypeError('Invalid type for learning rate')
        elif new_rate <= 0 or new_rate >= 1:
            raise ValueError('Invalid value for learning rate')
        else:
            self.__rate__ = new_rate

    def input(self, data):
        """Function takes first layer as vector and return last layer"""

        #checking input type
        if type(data) != np.ndarray:
            raise TypeError('Not a matrix!')

        #checking size
        shape = data.shape
        first_syn = self.__syn__[0]
        if len(shape) == 1 and shape[0] == first_syn.shape[1]:
            first_layer = data[:, np.newaxis]
        elif shape[0] == first_syn.shape[1] and shape[1] == 1:
            first_layer = data
        elif shape[1] == first_syn.shape[1] and shape[0] == 1:
            first_layer = data.T
        else:
            raise ValueError('Invalid size')

        #check 0 <= data[i, 0] <=1
        for i in range(first_layer.shape[0]):
            if first_layer[i, 0] < 0 or first_layer[i, 0] > 1:
                raise ValueError('Invalid vector value: ' + str(first_layer[i, 0]))

        #init first layer
        self.__layers__.clear()
        self.__layers__.append(first_layer)

        #calculate layers
        for i in range(len(self.__syn__)):
            self.__layers__.append(perceptron.__nonlin__(np.dot(self.__syn__[i], self.__layers__[-1]) + self.__biases__[i]))

        #return last layer
        return self.__layers__[-1]

    def tweak(self, result):
        """Backprapogapion"""

        #check layers and variable type
        if len(self.__layers__) == 0:
            raise ReferenceError('No saved layers')
        elif type(result) != np.ndarray:
            raise TypeError('Invalid type')

        #check matrix shape
        shape = result.shape
        last_layer = self.__layers__[-1]
        if len(shape) == 1 and shape[0] == last_layer.shape[0]:
            t_error = result[:, np.newaxis]
        elif shape == last_layer.shape:
            t_error = result
        elif shape == reversed(last_layer.shape):
            t_error = result.T
        else:
            raise ValueError('Invalid size')
        t_error -= last_layer

        #calculating deltas
        deltas = list()
        length = len(self.__syn__)
        deltas.insert(0, 2 * t_error * perceptron.__nonlin__(last_layer, deriv=True))
        for i in range(-1, -length + 1, -1):
            error = self.__syn__[i].T.dot(deltas[0])
            deltas.insert(0, 2 * error * perceptron.__nonlin(last_layer, deriv=True))

        # adjusting synapses
        for i in range(-1, -length, -1):
            self.__syn__[i] += (deltas[i].dot(self.__layers__[i - 1].T) * self.__rate__)

        return np.sum(np.square(t_error))


    def __nonlin__(x, deriv=False):
        """Nonlin function for calclulating value of neuron or derivative"""
        if deriv:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))
