import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e-4 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1) # TODO - What dimension should you pass to the softmax constructor?
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        # Q = Q.transpose(0, 1, 3, 2)  # 调整为 (batch_size, num_heads, depth, seq_len_q)
        # K = K.transpose(0, 1, 3, 2)  # 调整为 (batch_size, num_heads, depth, seq_len_k)
        self.Q = Q  # 存储 Q
        self.K = K  # 存储 K
        self.V = V  # 存储 V
            
        #attention_scores = np.matmul(Q, K.swapaxes(-2, -1)) # TODO - What is the shape of the result?
        d_k = K.shape[-1] # TODO - What is the shape of the result?
        #scaled_dot_product = attention_scores / np.sqrt(d_k) # TODO - What is the shape of the result?
        attention_scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            attention_scores = np.where(mask, float('-inf'), attention_scores)

        # Compute attention weights
        #attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        #attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
        attention_weights = self.softmax.forward(attention_scores)

        self.attention_scores = attention_weights  # Store attention scores for backward pass

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = np.matmul(self.attention_scores, V)

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = np.matmul(self.attention_scores.swapaxes(-2, -1), d_output)
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_weights = np.matmul(d_output, self.V.swapaxes(-2, -1))

        d_scores = self.softmax.backward(d_attention_weights)
        # Scale gradients by sqrt(d_k)
        d_scores = d_scores / np.sqrt(self.K.shape[-1])
        
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = np.matmul(d_scores, self.K)
        # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.matmul(d_scores.swapaxes(-2, -1), self.Q)
        print("d_scaled_dot_product:", d_scores)
        print("d_Q:", d_Q)
        print("d_K:", d_K)
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

