

from re import T

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow.nn import elu
from tensorflow import einsum
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
import tools
import numpy as np
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd

from tensorflow.python.ops.math_ops import to_int32

from tensorflow_probability.python.distributions.relaxed_onehot_categorical import RelaxedOneHotCategorical



class GroupLinearLayer(Layer):

    def __init__(self, units, nRIM, use_act=False, use_bias=True):
        super(GroupLinearLayer, self).__init__()
        self.units = units 
        self.nRIM = nRIM
        self.use_act = use_act
        self.use_bias = use_bias

    def gelu(self, x, approximate=False):
        if approximate:
            coeff = tf.cast(0.044715, x.dtype)
            return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
        else:
            return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

    def build(self, input_shape):
        # input_shape = (batch, [time,] nRIM, din)
        self.w = self.add_weight(name='group_linear_layer',
                                 shape=(self.nRIM, int(
                                     input_shape[-1]), self.units),
                                 initializer='random_normal',
                                 trainable=True, dtype='float32')
        if self.use_bias:
            self.b = self.add_weight(name='group_bias',
                                 shape=(self.nRIM, self.units),
                                 initializer='random_normal',
                                 trainable=True, dtype='float32')

    @tf.function                      
    def call(self, inputs):
        # out = tf.transpose(tf.matmul(tf.transpose(
        #     inputs, [1, 0, 2]), params), [1, 0, 2])
        out = tf.transpose(tf.matmul(tf.transpose(
            inputs, [1, 0, 2]), self.w), [1, 0, 2])
        if self.use_bias:
            out = out + self.b
        if self.use_act:
            out = tf.nn.elu(out)
            # out = self.gelu(out)
        return out


class ExterAttention(Layer):
    def __init__(self, dim, num_in_variables=None, num_out_variables=None, heads=8, 
    dim_head=64, softmax_type=None, group_wise=True):
        super(ExterAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.softmax_type = 'regulized'
        self.num_heads = heads
        self.num_in_variables = num_in_variables
        self.num_out_variables = num_out_variables

        self.scale = dim_head ** -0.5
        self.group_wise = group_wise

        # self.key_ = nn.Dense(units=self.num_heads*dim_head)
        # self.value_ = nn.Dense(units=self.num_heads*dim_head)
        # self.query_ = nn.Dense(units=self.num_heads*dim_head)

        self.key_ = GroupLinearLayer(units=self.num_heads*dim_head, nRIM=11, use_bias=False)
        self.value_ = GroupLinearLayer(units=self.num_heads*dim_head, nRIM=11, use_bias=False)
        self.query_ = GroupLinearLayer(units=self.num_heads*dim_head, nRIM=11, use_bias=False)


        self.to_out = [
                # nn.Dense(units=dim, activation=tf.nn.elu),
                GroupLinearLayer(dim, 11, use_act=True)
            ]

        self.to_out = Sequential(self.to_out)

    @tf.function
    def softmax_sms(self, attn, idx, pre_attn):
        N = attn.shape[-1]
        attn1 = tf.reduce_mean(attn, 1, keepdims=True)
        if idx != 0:
            pre_attn = (pre_attn * idx + attn1) / (idx + 1)
        else:
            pre_attn = attn1

        attn2 = tf.nn.softmax(pre_attn / 10, axis=-1)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = attn* attn2* N
        return attn, pre_attn

    @tf.function
    def call(self, pred_token, x, idx, pre_atten):
        query = self.query_(pred_token)
        key = self.key_(x)
        value = self.value_(x)

        key = tf.stack(tf.split(
            key, num_or_size_splits=self.num_heads, axis=-1), axis=1)
        value = tf.stack(tf.split(
            value, num_or_size_splits=self.num_heads, axis=-1), axis=1)
        query = tf.stack(tf.split(
            query, num_or_size_splits=self.num_heads, axis=-1), axis=1)

        attn = tf.matmul(
            query, key, transpose_b=True) * self.scale

        if self.softmax_type == 'regulized':
            attn, pre_atten = self.softmax_sms(attn, idx, pre_atten)
        else:
            attn = tf.nn.softmax(attn, axis=-1)
            
        context_layer = tf.matmul(attn, value)

        context_layer = tf.reshape(tf.transpose(context_layer, [0, 2, 1, 3]), 
                [tf.shape(x)[0], pred_token.shape[1], -1])

        return self.to_out(context_layer), pre_atten


class ExterBlock(tf.keras.layers.Layer):
    def __init__(
        self, dim, heads, dim_head, num_in_variables=None, 
        num_out_variables=None, softmax_type=None, group_wise=False):
        super().__init__()
        self.norm0 = nn.LayerNormalization()
        self.norm1 = nn.LayerNormalization()
        self.norm2 = nn.LayerNormalization()

        self.attn = ExterAttention(
            dim, heads=heads, dim_head=dim_head, num_in_variables=num_in_variables, 
            num_out_variables=num_out_variables, softmax_type=softmax_type, group_wise=group_wise)

        self._layer1 = Sequential(
            [   
                GroupLinearLayer(2*dim, 11, use_act=True),
                GroupLinearLayer(dim, 11, use_act=True)
                # GroupLinearLayer(dim, 11, use_act=True)
            ])

        # self._layer2 = Sequential(
        #     [   
        #         nn.Dense(units=dim),
        #         # nn.Dense(units=dim, activation=tf.nn.elu)
        #     ]) 
        
    @tf.function    
    def call(self, out_state_act, state_act, idx, pre_state_atten):
        n_state_act = self.norm0(state_act)

        out_state_act_, pre_state_atten = self.attn(
            self.norm1(out_state_act), n_state_act, idx=idx, 
            pre_atten=pre_state_atten)

        out_state_act = out_state_act_ + out_state_act
        
        out_state_act = self._layer1(self.norm2(out_state_act)) + out_state_act

        state_act = self._layer1(n_state_act) + state_act

        return state_act, out_state_act, pre_state_atten


class GRUCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, size, num_variables, norm=False, act=tf.tanh, update_scale=1, update_bias=-1):
        super().__init__()
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._update_scale = update_scale

        # self._layer = nn.Dense(3 * size, use_bias=False)

        self._layer = Sequential(
            [   
                nn.Dense(3 * size, activation=tf.nn.elu),
                # GroupLinearLayer(3 * size, 10, use_act=True),
                nn.Dense(3 * size, use_bias=False)
            ])
        # self._layer = GroupLinearLayer(
        #             units=3 * size, nRIM=num_variables)
            
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update * self._update_scale + self._update_bias)
        output = update * cand + (1 - update) * state
        return output


class CausalTransition(tools.Module):
    def __init__(self, dim, dim_deter, num_variables, deter_depth, post_depth, heads, post_heads, mlp_dim, dim_head=64, act=tf.nn.elu, emsenble=False, add_pos_embedding=False):
        super().__init__()
        self.num_variables = num_variables
        self.emsenble = emsenble
        # self.num_im_patch = 4
        # self.state_map = nn.Dense(units=dim, activation=act)
        self.dim = dim
        self.dim_deter = dim_deter
        self.action_map = nn.Dense(units=dim_deter)
        # self.map_in = GroupLinearLayer(units=dim_deter, nRIM=num_variables, use_act=True)

        self.add_pos_embedding = add_pos_embedding
        # if self.add_pos_embedding:
        self.pos_embedding = tf.Variable(
                initial_value=tf.random.normal([1, num_variables + 1, dim_deter]))

        self.out_state_act_embedding = tf.Variable(
                initial_value=tf.random.normal([1, num_variables, dim_deter]))

        # self.attn_sms = GroupAttention(
        #     dim_deter, num_variables=num_variables + 1, 
        #     heads=heads, dim_head=dim_head, softmax_type='sms')

        self.exter_atten_blocks = [ExterBlock(
            dim_deter, heads, dim_head, num_in_variables=num_variables+1, 
            num_out_variables=num_variables+1)
            for _ in range(4)]

        # self.mlp = GroupLinearLayer(units=3 * dim_deter, nRIM=num_variables)
        self.mlp1 = nn.Dense(3 * dim_deter, use_bias=False)
        # self.mlp2 = nn.Dense(dim_deter)
        self.mlp3 = nn.Dense(2 * dim_deter, use_bias=False)

        # self.rnn_cell0 = GroupGRUCell0(units=dim_deter, nRIM=num_variables)
        self.rnn_cell = GRUCell(dim_deter, num_variables, update_bias=-1.5)

        self.im_map = nn.Dense(units=200)
        # self.gate_post = nn.Dense(dim_deter, use_bias=False)
        # self.gate_post = GroupLinearLayer(
        #     units=2 * dim, nRIM=num_variables, use_bias=False)
        # self.gate_post = GroupLinearLayer(units=2, nRIM=num_variables)
        # self.proj_post = GroupLinearLayer(units=dim_deter, nRIM=num_variables) 

    @tf.function
    def __call__(
            self, stoch, deter, action, 
            first_state=False, return_more=False, action_embeding=False):
        b, dtype, n = deter.shape[0], deter.dtype, self.num_variables
        # state = tf.reshape(state, (b, n, -1))
        stoch, deter = tf.reshape(stoch, (b, n, -1)), tf.reshape(deter, (b, n, -1))
        state = self.rnn_cell(stoch, deter)
        # state = tf.concat([stoch, deter], -1)

        # reset, cand, update = tf.split(self.mlp2(state), 3, axis=-1)
        # reset = tf.nn.sigmoid(reset)
        # cand = tf.tanh(reset * cand)
        # update = tf.nn.sigmoid(update)
        # state = update * cand + (1 - update) * state

        if action_embeding != True:
            action_embed = self.action_map(action)
        else:
            action_embed = action

        state_act = tf.concat(
                [state, tf.reshape(action_embed, (b, 1, -1))], 1)

        # if self.add_pos_embedding and first_state:
        state_act = state_act + tf.cast(self.pos_embedding, dtype)
        out_state_act = tf.cast(self.out_state_act_embedding, dtype)
        out_state_act = state_act

        pre_state_atten = tf.zeros([], dtype=dtype)

        state_act_ = state_act
        for idx, blk in enumerate(self.exter_atten_blocks):
            state_act_, out_state_act, pre_state_atten = blk(
                out_state_act, state_act_, idx, pre_state_atten=pre_state_atten)

        # state_act_, state_act = tf.split(
        #     state_act_, [-1, 1], 1)[0], tf.split(state_act, [-1, 1], 1)[0]
        # parts = self.mlp1(tf.concat([state_act_, state_act], -1))
        # reset, cand, update = tf.split(parts, 3, axis=-1)
        # reset = tf.nn.sigmoid(reset)
        # cand = tf.tanh(reset * state_act + cand)
        # update = tf.nn.sigmoid(update)
        # state_act = update * cand + (1 - update) * state_act

        # out_state_act, state_act = self.mlp2(out_state_act), self.mlp2(state_act)

        # stop_out_state_act = tf.stop_gradient(out_state_act)
        # parts = self.mlp3(tf.concat([out_state_act, state_act], -1))
        # reset, update = tf.split(parts, 2, axis=-1)
        # reset, update = tf.reduce_mean(reset, -1, True), tf.reduce_mean(update, -1, True)
        # reset = tf.nn.sigmoid(reset)
        # cand = tf.tanh(out_state_act_)
        # update = tf.nn.sigmoid(update)
        # weight = update * reset
        # state_act = update * tf.tanh(out_state_act)  + (1 - update) * state_act
        
        state_act = out_state_act
        
        state = tf.split(state_act, [-1, 1], 1)[0]

        elbow_sms = (tf.reduce_max(pre_state_atten, -1, keepdims=True) 
            + tf.reduce_min(pre_state_atten, -1, keepdims=True)) * 1/2
        mask_state = pre_state_atten >= elbow_sms

        decision = tf.cast(mask_state, dtype=dtype)

        if return_more:
            return state, action_embed, decision
        else:
            return state

    @tf.function
    def get_post_embedding0(
            self, deter_embedding, im_embedding):
        n = self.num_variables
        b = deter_embedding.shape[0]
        
        deter_embedding = tf.reshape(deter_embedding, (b, n, -1))

        im_embedding = tf.reshape(self.im_map(im_embedding), (b, n, -1))

        post_embedding = self.im_map1(
            tf.concat([im_embedding, deter_embedding], -1)) 

        return post_embedding

    def get_post_embedding(self, im_embedding):

        return self.im_map(im_embedding)



if __name__ == '__main__':
    v = CausalTransition(
        dim=10,
        num_variables=8,
        depth=4,
        heads=4,
        mlp_dim=20
    )
    x = tf.random.normal(shape=[4000, 8, 10])
    a = tf.random.normal(shape=[4000, 1, 10])
    # e = tf.random.normal(shape=[4000, 1, 10])
    preds = v(x, a)


