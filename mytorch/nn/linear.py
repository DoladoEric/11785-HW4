import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.W = np.random.randn(out_features, in_features)  # TODO
        self.b = np.random.randn(out_features,1)  # TODO
    def init_weights(self, weight: np.ndarray, bias: np.ndarray):
        """
        Initialize the weights and biases of the Linear layer.

        :param weight: A numpy array of shape (out_features, in_features) representing the weights.
        :param bias: A numpy array of shape (1, out_features) representing the biases.
        """
        self.W = weight
        self.b = bias
    def __call__(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)
        """
        return self.forward(A)
    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.input_shape = A.shape
        A_flattened = A.reshape(-1, A.shape[-1])
        self.A = A_flattened  # TODO
        self.N = A_flattened.shape[0]  # - store the batch size parameter of the input A

        # Think how can `self.ones` help in the calculations and uncomment below code snippet.
        self.ones = np.ones((self.N, 1))

        Z = np.dot(A_flattened, self.W.T) + self.b  # TODO

        Z_unflattened = Z.reshape(*self.input_shape[:-1], -1)

        return Z_unflattened  
    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dLdZ_reshaped = dLdZ.reshape(-1, dLdZ.shape[-1])
        A_reshaped = self.A
        dLdA = np.dot(dLdZ_reshaped,self.W)  # TODO
        self.dLdW = np.dot(dLdZ_reshaped.T , A_reshaped)  # TODO
        #self.dLdb = np.dot(dLdZ.T,self.ones)  # TODO
        #self.dLdb = np.sum(dLdZ_reshaped, axis=0, keepdims=True)
        self.dLdb = np.sum(dLdZ_reshaped, axis=0, keepdims=True)  # Shape: (1, out_features)

        # Step 4: Reshape dLdA back to the original shape of input A
        dLdA_unflattened = dLdA.reshape(*self.input_shape)

        # Debugging (optional)
        if self.debug:
            self.dLdA = dLdA_unflattened

        # Return the gradient of loss wrt input A
        return dLdA_unflattened
