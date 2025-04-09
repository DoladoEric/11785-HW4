import numpy as np
import math
import scipy
import scipy.special


### No need to modify Identity class
class Identity:
    """
    Identity activation function.
    """

    def forward(self, Z):
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.Sigmoid!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Sigmoid Section) for further details on Sigmoid forward and backward expressions.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
    def backward(self, dldA) :
        dAdZ = self.A * (1 - self.A)
        dLdZ = dldA * dAdZ
        return dLdZ


class Tanh:
    """
    Tanh activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.Tanh!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Tanh Section) for further details on Tanh forward and backward expressions.
    """
    def forward(self, Z):
        exp_Z = np.exp(Z)
        exp_neg_Z = np.exp(-Z)
        self.A = (exp_Z - exp_neg_Z) / (exp_Z + exp_neg_Z)
        return self.A
    def backward(self, dLdA):
        dAdZ = 1 - self.A ** 2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.ReLU!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: ReLU Section) for further details on ReLU forward and backward expressions.
    """
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A
    def backward(self,dLdA):
        dLdZ = np.where(self.A > 0, dLdA, 0)
        return dLdZ


class GELU:
    """
    GELU (Gaussian Error Linear Unit) activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.GELU!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: GELU Section) for further details on GELU forward and backward expressions.
    Note: Feel free to save any variables from gelu.forward that you might need for gelu.backward.
    """
    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z*(1+scipy.special.erf(Z/math.sqrt(2)))
        return self.A
    def backward(self,dLdA):
        dLdZ = dLdA*(0.5*(1+scipy.special.erf(self.Z/math.sqrt(2)))+(self.Z/math.sqrt(2*math.pi))*np.exp(-self.Z*self.Z/2))
        return dLdZ


class Softmax:
    """
    Softmax activation function.

    ToDO:
    On same lines as above, create your own mytorch.nn.Softmax!
    Complete the 'forward' function.
    Complete the 'backward' function.
    Read the writeup (Hint: Softmax Section) for further details on Softmax forward and backward expressions.
    Hint: You read more about `axis` and `keep_dims` attributes, helpful for future homeworks too.
    """
    def __init__(self,dim = -1):
        self.dim = dim

    def forward(self, Z):
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        Note: How can we handle large overflow values? Hint: Check numerical stability.
        """
        # e_Z = np.exp(Z - np.max(Z, axis = 1, keepdims = True))
        # self.A = e_Z / np.sum(e_Z, axis = 1, keepdims = True)
        # self.N = Z.shape[0] * Z.shape(1)
        # self.C = Z.shape[2]
        Z_max = np.max(Z, axis = self.dim, keepdims=True)
        Z_stable = Z - Z_max

        exp_z = np.exp(Z_stable)

        sum_exp_z = np.sum(exp_z, axis=self.dim, keepdims=True)

        self.A = exp_z / sum_exp_z

        return self.A # TODO - What should be the return value?

    def backward(self, dLdA):
        # Calculate the batch size and number of features
                # Step 1: Move the specified dimension to the last position
        Z_shape = self.A.shape  # Original shape of the input Z
        moved_dLdA = np.moveaxis(dLdA, self.dim, -1)  # Move the specified dimension to the last position
        moved_A = np.moveaxis(self.A, self.dim, -1)  # Move the specified dimension to the last position

        # Step 2: Flatten the tensor to 2D (batch_size, num_classes)
        flattened_dLdA = moved_dLdA.reshape(-1, moved_A.shape[-1])  # Shape: (N · H · W, C)
        flattened_A = moved_A.reshape(-1, moved_A.shape[-1])  # Shape: (N · H · W, C)

        # Step 3: Initialize dLdZ with zeros
        dLdZ_flattened = np.zeros_like(flattened_dLdA)  # Shape: (N · H · W, C)

        # Step 4: Compute the Jacobian and gradient for each row
        for i in range(flattened_A.shape[0]):  # Iterate over each row
            a = flattened_A[i, :]  # Shape: (C,)
            jacobian = np.diag(a) - np.outer(a, a)  # Compute the Jacobian matrix (C, C)
            dLdZ_flattened[i, :] = np.dot(flattened_dLdA[i, :], jacobian)  # Gradient for this row

        # Step 5: Reshape dLdZ back to the original moved shape
        dLdZ_moved = dLdZ_flattened.reshape(moved_dLdA.shape)  # Shape: (N, H, W, C)

        # Step 6: Move the last dimension back to its original position
        dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)  # Restore the original shape of Z

        return dLdZ

        # Fill dLdZ one data point (row) at a time.

#print("Running Sucessfully!")