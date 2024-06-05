import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec
from causal_transiton import CausalTransition, GroupLinearLayer

import tools


class RSSM(tools.Module):

  def __init__(
      self, stoch=30, deter=200, hidden=200, layers_input=1, layers_output=1,
      shared=False, discrete=False, act=tf.nn.elu, mean_act='none',
      std_act='softplus', min_std=0.1, cell='keras', feat_mode='full'):
    super().__init__()
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._embed = None
    self.feat_mode = feat_mode
    self.feat_size = stoch + deter
    if cell == 'gru':
      self._cell = tfkl.GRUCell(self._deter)
    elif cell == 'gru_layer_norm':
      self._cell = GRUCell(self._deter, norm=True)
    else:
      raise NotImplementedError(cell)
    self.num_variables = 10
    
    self.causal_transition = CausalTransition(
        dim=(deter + stoch) // self.num_variables, 
        dim_deter=deter // self.num_variables, 
        num_variables=self.num_variables, deter_depth=5, 
        post_depth=1, heads=3, post_heads=4, mlp_dim=20, 
        dim_head=10, add_pos_embedding=False)
        
  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
            mean=tf.zeros([batch_size, self._stoch], dtype),
            std=tf.zeros([batch_size, self._stoch], dtype),
            stoch=tf.zeros([batch_size, self._stoch], dtype),
            deter=self._cell.get_initial_state(None, batch_size, dtype),
            decision=None,
            cd_deter_mean=None,
            cd_deter_std=None,
            action_embed=None,
                )
    return state

  @tf.function
  def observe(self, embed, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
        state = self.initial(tf.shape(action)[0])
    embed, action = swap(embed), swap(action)
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (action, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, action, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
        state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = swap(action)
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    stoch = state['stoch']
    if self._discrete:
        shape = stoch.shape[:-2] + [self._stoch * self._discrete]
        stoch = tf.reshape(stoch, shape)
    if self.feat_mode == 'stoch':
        return stoch
    else:
        return tf.concat([stoch, state['deter']], -1)

  def get_dist(self, state, dtype=None):
    if self._discrete:
        logit = state['logit']
        logit = tf.cast(logit, tf.float32)
        dist = tfd.Independent(tools.OneHotDist(logit), 1)
        if dtype != tf.float32:
            dist = tools.DtypeDist(dist, dtype or state['logit'].dtype)
    else:
        mean, std = state['mean'], state['std']
        if dtype:
            mean = tf.cast(mean, dtype)
            std = tf.cast(std, dtype)
        dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    if not self._embed:
        self._embed = embed.shape[-1]
    prior = self.img_step(prev_state, prev_action, None, sample)
    if self._shared:
        post = self.img_step(prev_state, prev_action, embed, sample)
    else:
        x = tf.concat([prior['deter'], embed], -1)
        for i in range(self._layers_output):
            x = self.get(f'obi{i}', tfkl.Dense, self._hidden, self._act)(x)
        stats = self._suff_stats_layer('obs', x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        # post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        post = {'stoch': stoch, 'deter': prior['deter'], **stats,
                'decision': prior['decision'], 
                'cd_deter_mean': prior['cd_deter_mean'], 'cd_deter_std': prior['cd_deter_mean'], 
                'action_embed': prior['action_embed']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, embed=None, sample=True):
    prev_stoch = prev_state['stoch']
    if self._discrete:
        shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
        prev_stoch = tf.reshape(prev_stoch, shape)
    if self._shared:
        if embed is None:
            shape = prev_action.shape[:-1] + [self._embed]
            embed = tf.zeros(shape, prev_action.dtype)
        x = tf.concat([prev_stoch, prev_action, embed], -1)
    else:
        x = tf.concat([prev_stoch, prev_action], -1)
    for i in range(self._layers_input):
        x = self.get(f'ini{i}', tfkl.Dense, self._hidden, self._act)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    for i in range(self._layers_output):
        x = self.get(f'imo{i}', tfkl.Dense, self._hidden, self._act)(x)
    stats = self._suff_stats_layer('ims', x)
    if sample:
        stoch = self.get_dist(stats).sample()
    else:
        stoch = self.get_dist(stats).mode()
    
    cd_deter, action_embed, decision = self.causal_transition(
            tf.stop_gradient(prev_state['stoch']), tf.stop_gradient(deter), 
            tf.stop_gradient(prev_action), return_more=True)
    
    x = self.get(f'cd_deter', tfkl.Dense, 2 * self._deter / self.num_variables)(cd_deter)
    cd_deter_mean, cd_deter_std = tf.split(x, 2, -1)
    cd_deter_std = tf.nn.sigmoid(cd_deter_std) + 0.01
    cd_deter_mean, cd_deter_std = tf.reshape(
        cd_deter_mean, [cd_deter_mean.shape[0], -1]), tf.reshape(cd_deter_std, [cd_deter_std.shape[0], -1])
            
    prior = {'stoch': stoch, 'deter': deter, **stats,
                'decision': decision, 
                'cd_deter_mean': cd_deter_mean, 'cd_deter_std': cd_deter_std, 
                'action_embed': action_embed}
    
    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
        x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
        logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
        return {'logit': logit}
    else:
        x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
        mean, std = tf.split(x, 2, -1)
        mean = {
            'none': lambda: mean,
            'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
        }[self._mean_act]()
        std = {
            'softplus': lambda: tf.nn.softplus(std),
            'abs': lambda: tf.math.abs(std + 1),
            'sigmoid': lambda: tf.nn.sigmoid(std),
            'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
        }[self._std_act]()
        std = std + self._min_std
        return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, balance, free, scale):
    kld = tfd.kl_divergence
    dist = lambda x: self.get_dist(x, tf.float32)
    if balance == 0.5:
        value = kld(dist(prior), dist(post))
        loss = tf.reduce_mean(tf.maximum(value, free))
    else:
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        value = kld(dist(prior), dist(sg(post)))
        pri = tf.reduce_mean(value)
        pos = tf.reduce_mean(kld(dist(sg(prior)), dist(post)))
        pri, pos = tf.maximum(pri, free), tf.maximum(pos, free)
        loss = balance * pri + (1 - balance) * pos
    loss *= scale
    return loss, value

class GC_Encoder(tools.Module):
  def __init__(
      self, depth=8, act=tf.nn.leaky_relu, kernels=(4, 4, 4, 4)):
    self._act = act
    self._depth = depth
    self._kernels = kernels

  def __call__(self, gc_obs):
    kwargs = dict(strides=2, activation=self._act)
    Conv = tfkl.Conv2D
    Bn   = tfkl.BatchNormalization
    x = tf.reshape(gc_obs, (-1,) + tuple(gc_obs.shape[-3:]))
    x = self.get('h1', Conv, 1 * self._depth, self._kernels[0], **kwargs)(x)
    x = self.get('h2', Conv, 2 * self._depth, self._kernels[1], **kwargs)(x)
    x = self.get('bn1', Bn, axis = -1)(x)
    x = self.get('h3', Conv, 4 * self._depth, self._kernels[2], **kwargs)(x)
    x = self.get('bn2', Bn, axis = -1)(x)
    x = self.get('h4', Conv, 8 * self._depth, self._kernels[3], **kwargs)(x)
    x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    shape = tf.concat([tf.shape(gc_obs)[:-3], [x.shape[-1]]], 0)
    return tf.reshape(x, shape)

class GC_Distance(tools.Module):

  def __init__(
      self, act=tf.nn.relu, layers = 4, units = 128, out_dim=1, input_type='feat', normalize_input=False):

    self._layers = layers
    self._num_layers = layers
    self._units = units
    self._act = act
    self.out_dim = out_dim
    
    self._input_type = input_type
    self._normalize_input = normalize_input

  def __call__(self, gc_obs, no_softmax=False):
    if self._normalize_input:
      _inp, _goal = tf.split(gc_obs, 2, axis = -1)
      _inp = _inp/(tf.norm(_inp)+1e-8)
      _goal = _goal/(tf.norm(_goal)+1e-8)
      x = tf.concat([_inp, _goal], axis=-1)
    else:
      x = gc_obs

    for index in range(self._layers):
      x = self.get(f'fc{index}', tfkl.Dense, self._units, self._act)(x)
      x = self.get(f'fc_bn{index}', tfkl.BatchNormalization, axis = -1)(x)

    out = tf.squeeze(self.get(f'hout', tfkl.Dense, self.out_dim)(x))
    if self.out_dim <= 1 or no_softmax:
      return out
    else:
      return tf.nn.softmax(out)


class GC_Critic(tools.Module):

  def __init__(
      self, act=tf.nn.relu, layers = 4, units = 128):

    self._layers = layers
    self._encoder = GC_Encoder()
    self._num_layers = layers
    self._units = units
    self._act = act

  def __call__(self, gc_obs, action):
   
    x = tf.concat([self._encoder(gc_obs), action], axis = -1)
    for index in range(self._layers):
   
      x = self.get(f'fc{index}', tfkl.Dense, self._units, self._act)(x)
      x = self.get(f'fc_bn{index}', tfkl.BatchNormalization, axis = -1)(x)

    return tf.squeeze(self.get(f'hout', tfkl.Dense, 1)(x))
    

class GC_Actor(tools.Module):

  def __init__(
      self, size, act=tf.nn.relu, layers = 10, units = 128, from_images=True):
    self._size = size
    self._layers = layers
    self._num_layers = layers
    self._units = units
    self._act = act
    self.from_images = from_images
    if from_images:
      self._encoder = GC_Encoder()

  def __call__(self, gc_obs):
    x = self._encoder(gc_obs) if self.from_images else gc_obs
    for index in range(self._layers):
   
      x = self.get(f'fc{index}', tfkl.Dense, self._units, self._act)(x)
      x = self.get(f'fc_bn{index}', tfkl.BatchNormalization, axis = -1)(x)

    x = self.get(f'hout', tfkl.Dense, self._size)(x)
    return tfkl.Activation('tanh')(x)

class ConvEncoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
    self._act = act
    self._depth = depth
    self._kernels = kernels
    self.embed_size = depth * 32

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    Conv = tfkl.Conv2D
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = self.get('h1', Conv, 1 * self._depth, self._kernels[0], **kwargs)(x)
    x = self.get('h2', Conv, 2 * self._depth, self._kernels[1], **kwargs)(x)
    x = self.get('h3', Conv, 4 * self._depth, self._kernels[2], **kwargs)(x)
    x = self.get('h4', Conv, 8 * self._depth, self._kernels[3], **kwargs)(x)
    x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
    return tf.reshape(x, shape)
  

class ConvEncoderWithState(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
    self._act = act
    self._depth = depth
    self._kernels = kernels
    self.embed_size = depth * 32 + 9

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    Conv = tfkl.Conv2D
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = self.get('h1', Conv, 1 * self._depth, self._kernels[0], **kwargs)(x)
    x = self.get('h2', Conv, 2 * self._depth, self._kernels[1], **kwargs)(x)
    x = self.get('h3', Conv, 4 * self._depth, self._kernels[2], **kwargs)(x)
    x = self.get('h4', Conv, 8 * self._depth, self._kernels[3], **kwargs)(x)
    x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
    return tf.concat([tf.reshape(x, shape), obs['state'][..., :9]], -1)


class ConvDecoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6),
      thin=True):
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin

  def __call__(self, features, dtype=None):
    kwargs = dict(strides=2, activation=self._act)
    ConvT = tfkl.Conv2DTranspose
    if self._thin:
      x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
      x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    else:
      x = self.get('h1', tfkl.Dense, 128 * self._depth, None)(features)
      x = tf.reshape(x, [-1, 2, 2, 32 * self._depth])
    x = self.get('h2', ConvT, 4 * self._depth, self._kernels[0], **kwargs)(x)
    x = self.get('h3', ConvT, 2 * self._depth, self._kernels[1], **kwargs)(x)
    x = self.get('h4', ConvT, 1 * self._depth, self._kernels[2], **kwargs)(x)
    x = self.get('h5', ConvT, self._shape[-1], self._kernels[3], strides=2)(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if dtype:
      mean = tf.cast(mean, dtype)
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseHead(tools.Module):

  def __init__(
      self, shape, layers, units, act=tf.nn.elu, dist='normal', std=1.0):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
        x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
    mean = tf.reshape(mean, tf.concat(
        [tf.shape(features)[:-1], self._shape], 0))
    if self._std == 'learned':
        std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
        std = tf.nn.softplus(std) + 0.01
        std = tf.reshape(std, tf.concat(
            [tf.shape(features)[:-1], self._shape], 0))
    else:
        std = self._std
    if dtype:
        mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
    if self._dist == 'normal':
        return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
    if self._dist == 'huber':
        return tfd.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
    if self._dist == 'binary':
        return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
    raise NotImplementedError(self._dist)


class ActionHead(tools.Module):

  def __init__(
      self, size, layers, units, act=tf.nn.elu, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    # assert min_std <= 2
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

  def __call__(self, *args, dtype=None):
    if len(args) == 1:
      x = args[0]
    else:
      x = tf.concat(args, -1)
    for index in range(self._layers):
      kw = {}
      if index == self._layers - 1 and self._outscale:
        kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
            self._outscale)
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, **kw)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = 5 * tf.tanh(mean / 5)
      std = tf.nn.softplus(std + 5) + 5
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'normal':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'normal_1':
      mean = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        mean = tf.cast(mean, dtype)
      dist = tfd.Normal(mean, 1)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'trunc_normal':
      # https://www.desmos.com/calculator/mmuvuhnyxo
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = tf.cast(x, tf.float32)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = 2 * tf.nn.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = tf.cast(x, tf.float32)
      dist = tools.OneHotDist(x, dtype=dtype)
      dist = tools.DtypeDist(dist, dtype)
    elif self._dist == 'onehot_gumble':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      temp = self._temp
      dist = tools.GumbleDist(temp, x, dtype=dtype)
    else:
      raise NotImplementedError(self._dist)
    return dist


class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]

def get_mlp_model(name, hidden_units, out_dim):
  with tf.name_scope(name) as scope:
    model = tfk.Sequential()
    for units in hidden_units:
        model.add(tfk.layers.Dense(units, activation='elu'))
    model.add(tfk.layers.Dense(out_dim, activation='tanh'))
  return model


class MlpBlock(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNormalization()
        self._layer1 = Sequential(
            [   
                # nn.Dense(units=3 * dim, activation=tf.nn.elu),
                GroupLinearLayer(3 * dim, 10, use_act=True),
                nn.Dense(3 * dim, use_bias=False)
            ])
            
    @tf.function      
    def call(self, x):
        parts = self._layer1(x)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = tf.tanh(reset * cand)
        update = tf.nn.sigmoid(update)
        x = update * cand + (1 - update) * x
        return x


class TSSM(RSSM):
    def __init__(self, *kwarg, **kwargs):
        super().__init__(*kwarg, **kwargs)
        dim_stoch, dim_deter = kwarg[:2]

        self.num_variables = 10
        self.causal_transition = CausalTransition(
            dim=(dim_deter + dim_stoch) // self.num_variables, 
            dim_deter=dim_deter // self.num_variables, 
            num_variables=self.num_variables, deter_depth=5, 
            post_depth=1, heads=3, post_heads=4, mlp_dim=20, 
            dim_head=20, add_pos_embedding=False)
        self._cell._layer = None

        units = int(dim_stoch / self.num_variables * 2) 
        # self.pri2stoch = tfkl.Dense(units=200)
        # self.pri2stoch = Sequential(
        #     [   
        #         tfkl.Dense(units=200, activation=tf.nn.elu),
        #         tfkl.Dense(units=100)
        #     ])

        self.post2stoch = tfkl.Dense(units=dim_stoch * 2)
        
        self.pri2stoch = [MlpBlock(30) for _ in range(1)]
        self.pri2stoch1 = Sequential(
            [
                tfkl.Dense(units=dim_stoch // self.num_variables * 2)
            ])
        self.state2stoch = tfkl.Dense(units=20)
        # self.pri2stoch = GroupLinearLayer(units=units, nRIM=self.num_variables)
        # self.post2stoch = GroupLinearLayer(units=units, nRIM=self.num_variables)

    def kl_loss(self, post, prior, balance, free, scale, free_avg=False):
        kld = tfd.kl_divergence
        def dist(x): return self.get_dist(x, tf.float32)
        if balance == 0.5:
            value = kld(dist(prior), dist(post))
            loss = tf.reduce_mean(tf.maximum(value, free))
        else:
            def sg(x): return tf.nest.map_structure(tf.stop_gradient, x)
            value_lhs = value = (kld(dist(prior), dist(sg(post))) 
                                        + kld(dist(sg(post)), dist(prior))) / 2

            value_rhs = (kld(dist(sg(prior)), dist(post))
                                + kld(dist(post), dist(sg(prior)))) / 2
            if free_avg:
                # pri = tf.reduce_mean(value_lhs)
                # pos = tf.reduce_mean(kld(dist(sg(prior)), dist(post)))
                pri, pos = tf.maximum(
                    tf.reduce_mean(value_lhs), free), tf.maximum(tf.reduce_mean(
                    value_rhs), free)
            else:
                pri, pos = tf.reduce_mean(
                    tf.maximum(value_lhs, free)), tf.reduce_mean(tf.maximum(
                    value_rhs, free))

            loss = balance * pri + (1 - balance) * pos

        # _value1 = tf.cast(
        #     tf.norm(tf.stop_gradient(post['deter1']) - prior['deter1'], 
        #     ord=2, axis=-1), tf.float32) 
        # _value2 = tf.cast(
        #     tf.norm(post['deter1'] - tf.stop_gradient(prior['deter1']), 
        #     ord=2, axis=-1), tf.float32) 

        # loss_mse = tf.reduce_mean(_value1 + _value2)
        # # value += value1
        # loss += loss_mse

        loss *= scale
        return loss, value


    def get_feat(self, state):
        stoch = state['stoch']
        if self.feat_mode == 'stoch':
            return stoch
        else:
            _shape = stoch.shape[:-1]
            shape = _shape + [self.num_variables]
            stoch = tf.reshape(stoch, (*shape, -1))
            deter = tf.reshape(state['deter'], (*shape, -1))
            return tf.reshape(tf.concat([stoch, deter], -1), (*_shape, -1))

    @tf.function
    def observe(self, embed, action, state=None):
        def swap(x): return tf.transpose(
            x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action)[0])
        embed, action = swap(embed), swap(action)
        first_state = tf.cast(tf.concat(
            [tf.ones(1), tf.zeros(embed.shape[0] - 1)], 0), bool) 

        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action, embed, first_state), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}

        return post, prior

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, first_state=False, sample=True):
        if not self._embed:
            self._embed = embed.shape[-1]
        prior = self.img_step(
            prev_state, prev_action, first_state, sample)
        deter = prior['deter']
        # x = self.causal_transition.get_post_embedding(embed)
        stats = self._suff_stats_layer(self.post2stoch(embed))
        # stats = self._suff_stats_layer(self.pri2stoch(x))

        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        # x = tf.reshape(x, (stoch.shape[0], -1))
        post = {'stoch': stoch, 'deter': deter, **stats, 'decision': prior['decision'], 'action_embed': prior['action_embed'], 'weight': prior['weight']}
        return post, prior

    @tf.function
    def img_step(
            self, prev_state, prev_action=None, first_state=False, sample=True, tuple_input=False):
        if tuple_input != True:
            prev_stoch = prev_state['stoch']
            B = prev_stoch.shape[0]
            deter, action_embed, decision, weight = self.causal_transition(
                prev_state['stoch'], prev_state['deter'], prev_action, 
                first_state=first_state, return_more=True)
        else:
            stoch, deter = prev_state
            B = stoch.shape[0]
            deter, action_embed, decision = self.causal_transition(
                deter, prev_action,
                first_state=first_state, return_more=True, action_embeding=True)

        # stats1 = self._suff_stats_layer(self.state2stoch(deter)) 
        # state_stoch = self.get_dist(stats1).sample()
        # state_stoch = tf.reshape(state_stoch, [B, -1])
        # stats = self._suff_stats_layer(self.pri2stoch(tf.reshape(deter, [B, -1])))
        for blk in self.pri2stoch:
            deter = blk(deter)
            
        stats = self._suff_stats_layer(self.pri2stoch1(deter))
        stats['mean'] = tf.reshape(stats['mean'], [B, -1])
        stats['std'] = tf.reshape(stats['std'], [B, -1])

        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        deter = tf.reshape(deter, (B, -1))
        prior = {'stoch': stoch, 'deter': deter, **stats, 
            'decision': decision, 'action_embed': action_embed, 'weight': weight}
        return prior

    def _suff_stats_layer(self, x):
        sh = x.shape[:-1]

        mean, std = tf.split(x, 2, -1)
        mean = {
            'none': lambda: mean,
            'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
        }[self._mean_act]()
        std = {
            'softplus': lambda: tf.nn.softplus(std),
            'abs': lambda: tf.math.abs(std + 1),
            'sigmoid': lambda: tf.nn.sigmoid(std),
            'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
            'sigmoid3': lambda: 3 * tf.nn.sigmoid(std / 3),
        }[self._std_act]()
        std = std + self._min_std

        return  {'mean': tf.reshape(mean, (*sh, -1)),
            'std': tf.reshape(std, (*sh, -1))}

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros(
                    [batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros(
                    [batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype),
                decision=None,
                action_embed=None,
                # state_stoch=tf.zeros([batch_size, 100], dtype),
                weight=None,
            )
        return state

class GC_Encoder(tools.Module):
    def __init__(
            self, depth=8, act=tf.nn.leaky_relu, kernels=(4, 4, 4, 4)):
        self._act = act
        self._depth = depth
        self._kernels = kernels
        self.embed_size = depth * 32

    def __call__(self, gc_obs):
        kwargs = dict(strides=2, activation=self._act)
        Conv = tfkl.Conv2D
        Bn = tfkl.BatchNormalization
        x = tf.reshape(gc_obs, (-1,) + tuple(gc_obs.shape[-3:]))
        x = self.get('h1', Conv, 1 * self._depth,
                     self._kernels[0], **kwargs)(x)
        x = self.get('h2', Conv, 2 * self._depth,
                     self._kernels[1], **kwargs)(x)
        x = self.get('bn1', Bn, axis=-1)(x)
        x = self.get('h3', Conv, 4 * self._depth,
                     self._kernels[2], **kwargs)(x)
        x = self.get('bn2', Bn, axis=-1)(x)
        x = self.get('h4', Conv, 8 * self._depth,
                     self._kernels[3], **kwargs)(x)
        x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
        shape = tf.concat([tf.shape(gc_obs)[:-3], [x.shape[-1]]], 0)
        return tf.reshape(x, shape)


class GC_Distance(tools.Module):

    def __init__(
            self, act=tf.nn.relu, layers=4, units=128, out_dim=1, input_type='feat', normalize_input=False):

        self._layers = layers
        self._num_layers = layers
        self._units = units
        self._act = act
        self.out_dim = out_dim

        self._input_type = input_type
        self._normalize_input = normalize_input

    def __call__(self, gc_obs, no_softmax=False):
        if self._normalize_input:
            _inp, _goal = tf.split(gc_obs, 2, axis=-1)
            _inp = _inp/(tf.norm(_inp)+1e-8)
            _goal = _goal/(tf.norm(_goal)+1e-8)
            x = tf.concat([_inp, _goal], axis=-1)
        else:
            x = gc_obs

        for index in range(self._layers):
            x = self.get(f'fc{index}', tfkl.Dense, self._units, self._act)(x)
            x = self.get(f'fc_bn{index}', tfkl.BatchNormalization, axis=-1)(x)

        out = tf.squeeze(self.get(f'hout', tfkl.Dense, self.out_dim)(x))
        if self.out_dim <= 1 or no_softmax:
            return out
        else:
            return tf.nn.softmax(out)


class GC_Critic(tools.Module):

    def __init__(
            self, act=tf.nn.relu, layers=4, units=128):

        self._layers = layers
        self._encoder = GC_Encoder()
        self._num_layers = layers
        self._units = units
        self._act = act

    def __call__(self, gc_obs, action):

        x = tf.concat([self._encoder(gc_obs), action], axis=-1)
        for index in range(self._layers):

            x = self.get(f'fc{index}', tfkl.Dense, self._units, self._act)(x)
            x = self.get(f'fc_bn{index}', tfkl.BatchNormalization, axis=-1)(x)

        return tf.squeeze(self.get(f'hout', tfkl.Dense, 1)(x))


class GC_Actor(tools.Module):

    def __init__(
            self, size, act=tf.nn.relu, layers=10, units=128, from_images=True):
        self._size = size
        self._layers = layers
        self._num_layers = layers
        self._units = units
        self._act = act
        self.from_images = from_images
        if from_images:
            self._encoder = GC_Encoder()

    def __call__(self, gc_obs):
        x = self._encoder(gc_obs) if self.from_images else gc_obs
        for index in range(self._layers):

            x = self.get(f'fc{index}', tfkl.Dense, self._units, self._act)(x)
            x = self.get(f'fc_bn{index}', tfkl.BatchNormalization, axis=-1)(x)

        x = self.get(f'hout', tfkl.Dense, self._size)(x)
        return tfkl.Activation('tanh')(x)


class ConvEncoder(tools.Module):

    def __init__(self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
        self._act = act
        self._depth = depth
        self._kernels = kernels
        self.embed_size = depth * 32

    def __call__(self, obs, use_bn=False):
        kwargs = dict(strides=2, activation=self._act)
        Conv = tfkl.Conv2D
        if use_bn:
            Bn = tfkl.BatchNormalization
        x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
        x = self.get('h1', Conv, 1 * self._depth,
                     self._kernels[0], **kwargs)(x)
        x = self.get('h2', Conv, 2 * self._depth,
                     self._kernels[1], **kwargs)(x)
        if use_bn:
            x = self.get('bn1', Bn, axis=-1)(x)
        x = self.get('h3', Conv, 4 * self._depth,
                     self._kernels[2], **kwargs)(x)
        if use_bn:
            x = self.get('bn2', Bn, axis=-1)(x)
        x = self.get('h4', Conv, 8 * self._depth,
                     self._kernels[3], **kwargs)(x)
        x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
        shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
        return tf.reshape(x, shape)


class ConvEncoderWithState(tools.Module):

    def __init__(
            self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
        self._act = act
        self._depth = depth
        self._kernels = kernels
        self.embed_size = depth * 32 + 9

    def __call__(self, obs):
        kwargs = dict(strides=2, activation=self._act)
        Conv = tfkl.Conv2D
        x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
        x = self.get('h1', Conv, 1 * self._depth,
                     self._kernels[0], **kwargs)(x)
        x = self.get('h2', Conv, 2 * self._depth,
                     self._kernels[1], **kwargs)(x)
        x = self.get('h3', Conv, 4 * self._depth,
                     self._kernels[2], **kwargs)(x)
        x = self.get('h4', Conv, 8 * self._depth,
                     self._kernels[3], **kwargs)(x)
        x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
        shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
        return tf.concat([tf.reshape(x, shape), obs['state'][..., :9]], -1)


class ConvDecoder(tools.Module):

    def __init__(
            self, depth=32, act=tf.nn.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6),
            thin=True):
        self._act = act
        self._depth = depth
        self._shape = shape
        self._kernels = kernels
        self._thin = thin

    def __call__(self, features, dtype=None):
        kwargs = dict(strides=2, activation=self._act)
        ConvT = tfkl.Conv2DTranspose
        if self._thin:
            features = self.get('h0', tfkl.Dense, 800, activation=self._act)(features)
            x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
            x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        else:
            x = self.get('h1', tfkl.Dense, 128 * self._depth, None)(features)
            x = tf.reshape(x, [-1, 2, 2, 32 * self._depth])
        x = self.get('h2', ConvT, 4 * self._depth,
                     self._kernels[0], **kwargs)(x)
        x = self.get('h3', ConvT, 2 * self._depth,
                     self._kernels[1], **kwargs)(x)
        x = self.get('h4', ConvT, 1 * self._depth,
                     self._kernels[2], **kwargs)(x)
        x = self.get(
            'h5', ConvT, self._shape[-1], self._kernels[3], strides=2)(x)
        mean = tf.reshape(x, tf.concat(
            [tf.shape(features)[:-1], self._shape], 0))
        if dtype:
            mean = tf.cast(mean, dtype)
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))


class DenseHead(tools.Module):

    def __init__(
            self, shape, layers, units, act=tf.nn.elu, dist='normal', std=1.0, stcoh=True):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._act = act
        self._dist = dist
        self._std = std
        self._stcoh = stcoh

    def __call__(self, features, dtype=None):
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
        mean = tf.reshape(mean, tf.concat(
            [tf.shape(features)[:-1], self._shape], 0))

        if self._stcoh:
            if self._std == 'learned':
                std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
                std = tf.nn.softplus(std) + 0.01
                std = tf.reshape(std, tf.concat(
                    [tf.shape(features)[:-1], self._shape], 0))
            else:
                std = self._std
            if dtype:
                mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
            if self._dist == 'normal':
                return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
            if self._dist == 'huber':
                return tfd.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
            if self._dist == 'binary':
                return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
            raise NotImplementedError(self._dist)
        else:
            return mean

class ActionHead(tools.Module):

    def __init__(
            self, size, layers, units, act=tf.nn.elu, dist='trunc_normal',
            init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
        # assert min_std <= 2
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._action_disc = action_disc
        self._temp = temp() if callable(temp) else temp
        self._outscale = outscale

    def __call__(self, *args, dtype=None):
        if len(args) == 1:
            x = args[0]
        else:
            x = tf.concat(args, -1)
        for index in range(self._layers):
            kw = {}
            if index == self._layers - 1 and self._outscale:
                kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
                    self._outscale)
            # x = self.get(f'h{index}', tfkl.Dense,
            #                 self._units, self._act, **kw)(x)
            if index == 0:
                x = self.get(f'h{index}', tfkl.Dense,
                            self._units, self._act, **kw)(x)
            else:
                x = self.get(f'h{index}', tfkl.Dense,
                            self._units, self._act, **kw)(x) + x
        if self._dist == 'tanh_normal':
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            if dtype:
                x = tf.cast(x, dtype)
            mean, std = tf.split(x, 2, -1)
            mean = tf.tanh(mean)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'tanh_normal_5':
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            if dtype:
                x = tf.cast(x, dtype)
            mean, std = tf.split(x, 2, -1)
            mean = 5 * tf.tanh(mean / 5)
            std = tf.nn.softplus(std + 5) + 5
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'normal':
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            if dtype:
                x = tf.cast(x, dtype)
            mean, std = tf.split(x, 2, -1)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.Independent(dist, 1)
        elif self._dist == 'normal_1':
            mean = self.get(f'hout', tfkl.Dense, self._size)(x)
            if dtype:
                mean = tf.cast(mean, dtype)
            dist = tfd.Normal(mean, 1)
            dist = tfd.Independent(dist, 1)
        elif self._dist == 'trunc_normal':
            # https://www.desmos.com/calculator/mmuvuhnyxo
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            x = tf.cast(x, tf.float32)
            mean, std = tf.split(x, 2, -1)
            mean = tf.tanh(mean)
            std = 2 * tf.nn.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.DtypeDist(dist, dtype)
            dist = tfd.Independent(dist, 1)
        elif self._dist == 'onehot':
            x = self.get(f'hout', tfkl.Dense, self._size)(x)
            x = tf.cast(x, tf.float32)
            dist = tools.OneHotDist(x, dtype=dtype)
            dist = tools.DtypeDist(dist, dtype)
        elif self._dist == 'onehot_gumble':
            x = self.get(f'hout', tfkl.Dense, self._size)(x)
            if dtype:
                x = tf.cast(x, dtype)
            temp = self._temp
            dist = tools.GumbleDist(temp, x, dtype=dtype)
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


def get_mlp_model(name, hidden_units, out_dim):
    with tf.name_scope(name) as scope:
        model = tfk.Sequential()
        for units in hidden_units:
            model.add(tfk.layers.Dense(units, activation='elu'))
        model.add(tfk.layers.Dense(out_dim, activation='tanh'))
    return model
