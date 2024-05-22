
import tensorflow.keras.layers as nn
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
import tools
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

        self.softmax_type = softmax_type
        self.num_heads = heads
        self.num_in_variables = num_in_variables
        self.num_out_variables = num_out_variables

        self.scale = dim_head ** -0.5
        self.group_wise = group_wise

        if group_wise:
            self.key_ = nn.Dense(units=self.num_heads*dim_head)
            self.value_ = nn.Dense(units=self.num_heads*dim_head)
            self.query_ = nn.Dense(units=self.num_heads*dim_head)     
        else:
            self.key_ = nn.Dense(units=self.num_heads*dim_head)
            self.value_ = nn.Dense(units=self.num_heads*dim_head)
            self.query_ = nn.Dense(units=self.num_heads*dim_head)

        self.to_out = [
                nn.Dense(units=dim, activation=tf.nn.elu),
            ]

        self.to_out = Sequential(self.to_out)

    def softmax_sms(self, attn, idx, pre_attn=None, eps=1e-6):
        N = attn.shape[-1]
        attn1 = tf.reduce_mean(attn, 1, keepdims=True)
        if idx != 0:
            pre_attn = (pre_attn * idx + attn1) / (idx + 1)
        else:
            pre_attn = attn1

        attn2 = tf.nn.softmax(pre_attn, axis=-1)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = attn* attn2* N
        return attn, pre_attn


    def call(self, pred_token, x, mask=None):
        N = x.shape[1]
        query = self.query_(pred_token)
        key = self.key_(x)
        value = self.value_(x)

        key = tf.stack(tf.split(
            key, num_or_size_splits=self.num_heads, axis=-1), axis=-3)
        value = tf.stack(tf.split(
            value, num_or_size_splits=self.num_heads, axis=-1), axis=-3)
        query = tf.stack(tf.split(
            query, num_or_size_splits=self.num_heads, axis=-1), axis=-3)

        attn = tf.matmul(
            query, key, transpose_b=True) * self.scale

        eps=1e-6
        # min_att = tf.reduce_min(attn, axis=-1, keepdims=True)
        # attn = attn - min_att
        # for stable training
        attn = tf.exp(attn) * mask
        attn = (attn + eps/N) / \
            (tf.reduce_sum(attn, axis=-1, keepdims=True) + eps)

        context_layer = tf.matmul(attn, value)
        if len(context_layer.shape) == 5:
            context_layer = tf.reshape(tf.transpose(context_layer, [0, 1, 3, 2, 4]), 
                [*pred_token.shape[:-1], -1])      
        else:
            context_layer = tf.reshape(tf.transpose(context_layer, [0, 2, 1, 3]), 
                [tf.shape(x)[0], pred_token.shape[1], -1])

        return self.to_out(context_layer)



class ExterBlock(tf.keras.layers.Layer):
    def __init__(
        self, dim, heads, dim_head, num_in_variables=None, 
        num_out_variables=None, softmax_type=None, group_wise=False):
        super().__init__()
        self.norm0 = nn.LayerNormalization()
        self.norm1 = nn.LayerNormalization()
        self.norm2 = nn.LayerNormalization()

        # self.norm4(tf.keras.Input(shape=(num_out_variables + 2, dim)))
        # self.norm7(tf.keras.Input(shape=(num_out_variables + 1, dim))) 

        self.attn = ExterAttention(
            dim, heads=heads, dim_head=dim_head, num_in_variables=num_in_variables, 
            num_out_variables=num_out_variables, softmax_type=softmax_type, group_wise=group_wise)

        # self.out_state_mlp = nn.Dense(units=dim, activation=tf.nn.elu)
        # self.out_action_mlp = nn.Dense(units=dim, activation=tf.nn.elu)
        # self.state_mlp = nn.Dense(units=dim, activation=tf.nn.elu)

        self.mlp0 = Sequential(
            [   
                # nn.Dense(units=2*dim, activation=tf.nn.elu),
                # nn.Dense(units=dim, activation=tf.nn.elu),
                GroupLinearLayer(dim, 10, use_act=True)
            ])

        self.mlp1 = Sequential(
            [   
                # nn.Dense(units=2*dim, activation=tf.nn.elu),
                # nn.Dense(units=dim, activation=tf.nn.elu),
                GroupLinearLayer(dim, 11, use_act=True)
            ])

    def call(self, out_state, state_action, mask):
        norm_state_action = self.norm0(state_action)

        out_state = self.attn(
            self.norm1(out_state), norm_state_action, mask=mask) + out_state

        out_state = self.mlp0(self.norm2(out_state)) + out_state
        state_action = self.mlp1(norm_state_action) + state_action

        return out_state, state_action


class CondSampler(tools.Module):
    def __init__(self, dim, dim_deter, num_variables, deter_depth, heads, dim_head=64):
        super().__init__()
        self.num_variables = num_variables
        self.dim = dim
        self.dim_deter = dim_deter

        self.out_state_token = tf.Variable(
            tf.random.normal([1, num_variables, dim]))

        self.out_pos_embedding = tf.Variable(
                initial_value=tf.random.normal([1, num_variables, dim]))

        heads = 3
        dim_head = 20
        self.exter_atten_blocks = [ExterBlock(
            dim, heads, dim_head, num_in_variables=num_variables+1, 
            num_out_variables=num_variables+1)
            for _ in range(2)]

        self.stoch = nn.Dense(units=2 * dim)
        self.norm = nn.LayerNormalization()

    @tf.function
    def __call__(
            self, B, L, state, action_embed, mask_state=None, evaluate=False, num_samples=10):
        b, dtype, n = state.shape[0], state.dtype, self.num_variables
        state = tf.reshape(state, (b, n, -1))

        out_state = tf.repeat(tf.cast(self.out_state_token, dtype), b, 0) 

        state_act = tf.concat([state, tf.expand_dims(action_embed, 1)], -2)

        out_state = out_state + tf.cast(self.out_pos_embedding, dtype)

        for blk in self.exter_atten_blocks:
            out_state, state_act = blk(out_state, state_act, mask_state)
        
        out_state = self.norm(out_state)
        
        feat = tf.reshape(state, [B, L, n, -1])

        if evaluate:
            stoch = tf.reshape(
                self.stoch(out_state), [B, L, n, -1])
            mean, std = tf.split(stoch, 2, -1)
            std = 2 * tf.nn.sigmoid(std / 2) + 0.01
            dist_state = tfd.MultivariateNormalDiag(mean[:-1], std[:-1]) 
            log_prob_state = dist_state.log_prob(feat[1:])
        else:
            mean, std = tf.split(self.stoch(out_state), 2, -1)
            std = 2 * tf.nn.sigmoid(std / 2) + 0.01
            mean, std = tf.reshape(
                mean, [B, L, n, -1]), tf.reshape(std, [B, L, n, -1])

            dist_state = tfd.MultivariateNormalDiag(mean[:, :-1], std[:, :-1])
            log_prob_state = dist_state.log_prob(feat[:, 1:])

        loss_state = - tf.reduce_mean(log_prob_state)

        if evaluate:
            return log_prob_state
        else:
            return loss_state
