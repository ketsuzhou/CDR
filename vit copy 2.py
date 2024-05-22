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

from tensorflow.python.ops.math_ops import to_int32

from tensorflow_probability.python.distributions.relaxed_onehot_categorical import RelaxedOneHotCategorical


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(
            tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Predictor(Layer):
    def __init__(self, embed_dim=384, predictor_type=None, num_variables=None):
        super().__init__()
        self.norm = nn.LayerNormalization()
        self.predictor_type = predictor_type
        if predictor_type == 'sms':
            self.layer = Sequential([
                nn.Dense(units=embed_dim, activation=tf.nn.relu),
                nn.Dense(units=embed_dim, activation=tf.nn.relu),
                # nn.Dense(units=embed_dim, activation=tf.nn.relu),
                nn.Dense(units=2 * num_variables, activation=tf.nn.relu),
                nn.Softmax()
            ])
        elif predictor_type == 'sts':
            self.layer = Sequential([
                nn.Dense(units=embed_dim, activation=tf.nn.relu),
                nn.Dense(units=embed_dim // 2, activation=tf.nn.relu),
                # nn.Dense(units=embed_dim // 4, activation=tf.nn.relu),
                nn.Dense(units=2, activation=tf.nn.relu),

            ])
        else:
            NotImplementedError

    def call(self, x, policy):
        x = self.norm(x)
        B, N, C = x.shape
        local_x = x[:, :, :C//2]
        if self.predictor_type == 'sms':
            global_x = einsum(
                'bij, bjd -> bid', policy, x[:, :, C//2:]) / tf.reduce_sum(policy, axis=-1, keepdims=True)
            x = tf.concat([local_x, global_x], axis=-1)
        elif self.predictor_type == 'sts':
            policy = tf.reshape(policy, (B, -1, 1))
            global_x = tf.reduce_sum(
                x[:, :, C//2:] * policy, axis=1, keepdims=True) / tf.reduce_sum(policy, axis=1, keepdims=True)
            x = tf.concat(
                [local_x, tf.repeat(global_x, repeats=N, axis=1)], axis=-1)
        else:
            NotImplementedError
        return self.layer(x)


class MLP(Layer):
    def __init__(self, dim, hidden_dim):
        super(MLP, self).__init__()

        self.net = Sequential([
            # nn.Dense(units=hidden_dim, activation=tf.nn.relu)
            nn.Dense(units=dim, activation=tf.nn.relu)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)


class Block(Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim, softmax_type=None):
        super().__init__()

        self.norm1 = nn.LayerNormalization()
        self.norm2 = nn.LayerNormalization()

        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, softmax_type=softmax_type)
        self.mlp = MLP(dim, mlp_dim)

    def call(self, x, training=True, policy=None):
        x = self.attn(self.norm1(x), training=training, policy=policy) + x
        x = self.mlp(self.norm2(x), training=training) + x
        return x


class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, softmax_type=None):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.softmax_type = softmax_type
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)

        if project_out:
            self.to_out = [
                nn.Dense(units=dim)]
        else:
            self.to_out = []

        self.to_out = Sequential(self.to_out)

    def softmax_sms(self, attn, policy, eps=1e-6):
        policy = tf.expand_dims(policy, axis=1)
        B, H, N, N = attn.shape

        max_att = tf.reduce_max(attn, axis=-1, keepdims=True)
        attn = attn - max_att

        # for stable training
        attn = tf.exp(attn) * policy
        attn = (attn + eps/N) / \
            (tf.reduce_sum(attn, axis=-1, keepdims=True) + eps)
        return attn

    def softmax_sts(self, attn, policy, eps=1e-6):
        B, H, N, N = attn.shape
        policy_row = tf.reshape(policy, (B, 1, N, 1))
        policy_col = tf.reshape(policy, (B, 1, 1, N))
        ones = tf.ones((1, 1, N, N))
        eye = tf.reshape(tf.eye(N), (1, 1, N, N))
        attn_policy = (ones - eye) * policy_row * policy_col + eye

        max_att = tf.reduce_max(attn, axis=-1, keepdims=True)
        attn = attn - max_att

        # for stable training
        attn = tf.exp(attn) * attn_policy
        attn = (attn + eps/N) / \
            (tf.reduce_sum(attn, axis=-1, keepdims=True) + eps)
        return attn

    def call(self, x, training=True, policy=None):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.softmax_type is None:
            attn = self.attend(dots)
        elif self.softmax_type == 'sms':
            attn = self.softmax_sms(dots, policy)
        elif self.softmax_type == 'sts':
            attn = self.softmax_sts(dots, policy)
        else:
            NotImplementedError

        # x = tf.matmul(attn, v)
        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x, training=training)

        return x


class Transition(tools.Module):
    def __init__(self, dim, num_variables, depth, heads, mlp_dim, dim_head=64):
        super().__init__()
        self.pos_embedding = tf.Variable(
            initial_value=tf.random.normal([1, num_variables + 2, dim]))

        self.sms_predictor = [Predictor(
            embed_dim=dim, predictor_type='sms', num_variables=num_variables)
            for _ in range(depth)]

        self.sts_predictor = [Predictor(
            embed_dim=dim, predictor_type='sts')
            for _ in range(depth)]
        
        self.deter_blocks = [Block(
            dim, heads, dim_head, mlp_dim, softmax_type='sms')
            for _ in range(depth)]

        self.post_blocks = [Block(
            dim, heads, dim_head, mlp_dim, softmax_type='sts')
            for _ in range(depth)]
        # self.GumbelSoftmax = GumbelSoftmax()

    def __call__(self, state, action, next_im_embedding, training=True):
        b, n, d = state.shape

        embedding = tf.concat([action, state], axis=1)
        embedding += self.pos_embedding[:, :(n + 1)]

        prev_sms_decision = tf.ones([b, n + 1, n + 1])
        prev_sts_decision = tf.ones([b, n + 1])
        action_policy = tf.reshape(tf.ones(b), (b, 1, 1))
        sms_decisions = []
        sts_decisions = []
        for i, blk in enumerate(self.deter_blocks):
            sms_score = tf.reshape(self.sms_predictor[i](
                embedding, prev_sms_decision), shape=(b, n + 1, -1, 2))

            sms_decision = gumbel_softmax(
                logits=sms_score, temperature=1, hard=True)[:, :, :, 0] * prev_sms_decision
            

            sms_decisions.append(sms_decision)
            
            sts_score = self.sts_predictor[i](embedding, prev_sts_decision)
            sts_decision = gumbel_softmax(
                logits=sts_score, temperature=1, hard=True)[:, :, 0:1]
            
            sts_decision = tf.concat(
                [action_policy, sts_decision[:, 1:]], axis=1)
            sts_decisions.append(sts_decision)
            
            decision = sms_decision * sts_decision
            
            decision = (tf.ones([n + 1, n + 1]) - tf.eye(n + 1)) * decision \
                + tf.eye(n + 1)
                
            embedding = blk(embedding, policy=decision)
            prev_sms_decision = decision
            
        sts_decision = sts_decisions[0]
        
        next_im_embedding += self.pos_embedding[:, -1]
        prev_sts_decision = tf.ones([b, n + 2])

        im_policy = tf.reshape(tf.ones(b), (b, 1))

        embedding = tf.concat([next_im_embedding, embedding], axis=1)
        for i, blk in enumerate(self.post_blocks):
            pred_score = self.post_sts_predictor[i](embedding, prev_sts_decision)

            hard_keep_decision = gumbel_softmax(
                logits=pred_score[:, :, 0], temperature=1,
                hard=True) * prev_sts_decision

            decision = tf.concat(
                [im_policy, hard_keep_decision[:, 1:]], axis=1)
            sts_decisions.append(decision)

            embedding = blk(embedding, policy=decision)
            prev_sts_decision = decision

        return embedding


if __name__ == '__main__':
    v = Transition(
        dim=20,
        num_variables=10,
        depth=4,
        heads=4,
        mlp_dim=20
    )
    x = tf.random.normal(shape=[4, 9, 20])
    a = tf.random.normal(shape=[4, 1, 20])
    e = tf.random.normal(shape=[4, 1, 20])
    preds = v(x, a, e)
