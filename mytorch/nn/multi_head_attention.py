from .linear import Linear
from .activation import Softmax
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()
        
        # Initialize your linear layer
        #  embed_dim -> embed_dim
        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        
        # TODO: Implement forward pass

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project the query, key, and value inputs into query, key, and value
        # (N, L, E) -> (N, L, embed_dim)
        q = self.q_proj(query)
        # (N, S, E) -> (N, S, embed_dim)
        k = self.k_proj(key)
        # (N, S, E) -> (N, S, embed_dim)
        v = self.v_proj(value)

        # Split the query, key, and value into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        q = q.reshape(self.N, self.L, self.num_heads, self.E // self.num_heads).transpose(0,2,1,3)
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        k = k.reshape(self.N, self.S, self.num_heads,self.E // self.num_heads).transpose(0,2,1,3)
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        v = v.reshape(self.N, self.S, self.num_heads,self.E // self.num_heads).transpose(0,2,1,3)

        # Merge the masks
        # (N, S) + (L, S) -> (N, H, L, S)
        # key_padding_mask =key_padding_mask.unsqueeze(1).unsqueeze(2)
        # attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        # mask = key_padding_mask | attn_mask
        key_padding_mask = np.expand_dims(np.expand_dims(key_padding_mask, axis=1), axis=2)
        attn_mask = np.expand_dims(np.expand_dims(attn_mask, axis=0), axis=1)
        mask = key_padding_mask | attn_mask
        # Apply the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads)
        # attn_outputs = np.matmul(q,k.swapaxes(-2,-1))
        # d_k = q.shape[-1]
        # attn_scores = attn_outputs / np.sqrt(d_k)  # scaled
        # # Apply the mask using np.where
        # attn_scores = np.where(mask == 1, float('-inf'), attn_scores)
        # softmax = Softmax(dim=-1)
        # attn_weights = softmax.forward(attn_scores)  # (N, H, L, S)
        # #attn_output = np.matmul(attn_weights, v)
        # attn_output = np.matmul(attn_weights, v)  # (N, H, L, embed_dim // num_heads)
        attn_output = self.attention.forward(q, k, v, mask)  # (N, num_heads, L, embed_dim // num_heads)
        # Merge the attention outputs   
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)
        attn_output = attn_output.swapaxes(1, 2).reshape(self.N, self.L, self.E) 

        # Project the attention outputs
        # (N, L, embed_dim) -> (N, L, embed_dim)
        output = self.out_proj(attn_output)
        print("this is the test")

        # Return output
        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """

        # TODO: Implement backward pass 

        # Backpropagate through the output projection   
        # (N, L, embed_dim) -> (N, L, embed_dim) 
        #d_attn_output = NotImplementedError
        d_attn_output = self.out_proj.backward(d_output)

        # Split the gradients into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        #d_attn_outputs =  d_attn_output.view(self.N, self.L, self.num_heads, self.E // self.num_heads).transpose(1, 2)
        
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)    
        d_attn_outputs = d_attn_output.reshape(self.N, self.L, self.num_heads, self.E // self.num_heads).swapaxes(1, 2)
        # Backpropagate through the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)

        # Merge the gradients
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)    
        d_q = self._concat_heads(d_q)
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_k = self._concat_heads(d_k)
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_v = self._concat_heads(d_v)

        # Backpropagate through the input projections   
        # (N, L, embed_dim) -> (N, L, embed_dim)
        d_q = self.q_proj.backward(d_q)
        # (N, S, embed_dim) -> (N, S, embed_dim)
        d_k =  self.k_proj.backward(d_k)
        # (N, S, embed_dim) -> (N, S, embed_dim)
        d_v = self.v_proj.backward(d_v)

        # Return gradients d_q, d_k, d_v
        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        # TODO: Implement merge masks
        device = key_padding_mask.device
        attn_mask = attn_mask.to(device)
        # Expand key_padding_mask to (N, 1, 1, S) and broadcast to (N, H, L, S)
        key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.expand(self.N, self.num_heads, self.L, self.S)
        
        # Expand attn_mask to (1, 1, L, S) and broadcast to (N, H, L, S)
        attention_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(self.N, self.num_heads, self.L, self.S)
        
        # Combine masks using logical_or - if either mask is True, we want to mask that position
        combined_mask = key_mask | attention_mask
        
        # Return combined mask
        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        # TODO: Implement split heads

        # Reshape: (N, L, embed_dim) -> (N, L, num_heads, embed_dim // num_heads)
        x = x.view(self.N, self.L, self.num_heads, self.E // self.num_heads)
        
        # Transpose: (N, L, num_heads, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        x = x.transpose(1, 2)
        
        # Return x
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        # TODO: Implement concat heads
        # Transpose: (N, num_heads, L, embed_dim // num_heads) -> (N, L, num_heads, embed_dim // num_heads)
        x = x.swapaxes(1, 2)
        
        # Reshape: (N, L, num_heads, embed_dim // num_heads) -> (N, L, embed_dim)
        x = x.reshape(self.N, self.L, self.E)
        
        # Return x
        return x
