import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

import models
import networks
import tools


class Random(tools.Module):

  def __init__(self, config):


    self._config = config
    self._float = prec.global_policy().compute_dtype

  def actor(self, feat, *args):
    shape = feat.shape[:-1] + [self._config.num_actions]
    if self._config.actor_dist == 'onehot':
      return tools.OneHotDist(tf.zeros(shape))
    else:
      ones = tf.ones(shape, self._float)
      return tfd.Uniform(-ones, ones)

  def train(self, start, feat, embed, kl):
    return None, {}


class Plan2Explore(tools.Module):

  def __init__(self, config, world_model, reward=None):

    self._config = config
    self._reward = reward
    self._behavior = models.ImagBehavior(config, world_model)
    self.actor = self._behavior.actor
    size = {
        'embed': 32 * config.cnn_depth,
        'stoch': config.dyn_stoch,
        'deter': config.dyn_deter,
        'feat': config.dyn_stoch + config.dyn_deter,
    }[self._config.disag_target]
    kw = dict(
        shape=size, layers=config.disag_layers, units=config.disag_units,
        act=config.act)
    self._networks = [
        networks.DenseHead(**kw) for _ in range(config.disag_models)]
    self._opt = tools.Optimizer(
        'ensemble', config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt)

  def train(self, start, feat, embed, kl):
    metrics = {}
    target = {
        'embed': embed,
        'stoch': start['stoch'],
        'deter': start['deter'],
        'feat': feat,
    }[self._config.disag_target]
    metrics.update(self._train_ensemble(feat, target))
    metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
    return None, metrics

  def _intrinsic_reward(self, feat, state, action):
    preds = [head(feat, tf.float32).mean() for head in self._networks]
    disag = tf.reduce_mean(tf.math.reduce_std(preds, 0), -1)
    if self._config.disag_log:
      disag = tf.math.log(disag)
    reward = self._config.expl_intr_scale * disag
    if self._config.expl_extr_scale:
      reward += tf.cast(self._config.expl_extr_scale * self._reward(
          feat, state, action), tf.float32)
    return reward

  def _train_ensemble(self, inputs, targets):

    if self._config.disag_offset:
      targets = targets[:, self._config.disag_offset:]
      inputs = inputs[:, :-self._config.disag_offset]
    targets = tf.stop_gradient(targets)
    inputs = tf.stop_gradient(inputs)
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self._networks]
      likes = [tf.reduce_mean(pred.log_prob(targets)) for pred in preds]
      loss = -tf.cast(tf.reduce_sum(likes), tf.float32)
    metrics = self._opt(tape, loss, self._networks)
    return metrics

  def act(self, feat, *args):
    return self.actor(feat)


from cond_sampler import CondSampler

class Causal_Plan2Explore(Plan2Explore):
    def __init__(self, config, world_model=None, reward=None):
        super().__init__(config, world_model, reward=reward)
        self._behavior = models.CausalImagBehavior(config, world_model)
        self.actor = self._behavior.actor

        self.disag_models = config.disag_models
        self.dynamics = world_model.dynamics

        # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        # tf.keras.mixed_precision.experimental.set_policy(policy)

        self.num_variables = 10
        self.cond_sampler = [CondSampler(
            dim=config.dyn_deter // self.num_variables, 
            dim_deter=config.dyn_deter // self.num_variables, 
            num_variables=self.num_variables, deter_depth=6, 
            heads=6, dim_head=20)
            for _ in range(config.disag_models)]
        dtype=prec.global_policy().compute_dtype
        # [head(2, 3, tf.ones([6, 300], dtype=dtype), tf.ones([6, 30], dtype=dtype), tf.zeros([6, 1,10, 11], dtype=dtype)) for head in self.cond_sampler]
        # tf.keras.mixed_precision.experimental.set_policy('float32')

        self._opt = tools.Optimizer(
            'ensemble', config.ensemble_model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt)

        self._random_mask = config.random_mask
        # self.Normalizer = Normalizer()
        self._updates = tf.Variable(0, trainable=False)

    # @tf.function
    def train(self, start, feat, embed, kl):
        metrics = {}

        metrics.update(self._train_ensemble(start, feat))
        metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])

        return None, metrics

    # @tf.function
    def _train_ensemble(self, start, feat):
        feat = start['deter']
        b, l = feat.shape[:2]
        # action = tf.reshape(action, [-1, action.shape[-1]])
        feat = tf.reshape(feat, [-1, feat.shape[-1]])
        mask_state = tf.reshape(start['decision'], [-1, *start['decision'].shape[-2:]]) 
        mask_state = tf.expand_dims(tf.split(mask_state, [-1, 1], 1)[0], 1) 
        action_embed = tf.reshape(start['action_embed'], [-1, start['action_embed'].shape[-1]]) 

        loss = tf.constant([0], dtype=tf.float32)
        for i, head in enumerate(self.cond_sampler):
            if self._updates % self.disag_models == i:
                with tf.GradientTape() as tape:
                    loss = head(b, l, feat, action_embed, mask_state)
                    loss = tf.cast(loss, tf.float32)
                self._opt(tape, loss, head)
                self._updates.assign_add(1)

        metrics = {}
        metrics['loss_state_mask'] = loss
        # metrics['loss_action_mask'] = loss_action_mask
        return metrics


    @tf.function
    def _intrinsic_reward(self, feat, state, action):
        feat = state['deter']
        h, bl = feat.shape[:2]

        feat1 = tf.reshape(feat, (h* bl, self.num_variables, -1))
        action1 = tf.reshape(state['action_embed'], (h* bl, -1))
        mask_state = tf.reshape(state['decision'], [-1, *state['decision'].shape[-2:]]) 
        mask_state = tf.expand_dims(tf.split(mask_state, [-1, 1], 1)[0], 1) 
        # num_samples = 10
        # log_prob_state_mask = self.cond_sampler(
        #     h, bl, feat1, action1, mask_state, evaluate=True, num_samples=num_samples)

        log_prob_state_mask = tf.concat([
            head(h, bl, feat1, action1, mask_state, evaluate=True)[None] for head in self.cond_sampler], axis=0) 

        # mask_state = tf.repeat(
        #     tf.cast(tf.ones([11, 11]) - tf.eye(11), 'float16')[:-1][None][None], 
        #     mask_state.shape[0], 0) 
        # log_prob_state_mask1 = self.cond_sampler(
        #     h, bl, feat1, action1, mask_state, evaluate=True, num_samples=num_samples)


        log_prob_state_mask = tf.reduce_logsumexp(
            tf.cast(log_prob_state_mask, tf.float32) + tf.math.log(1 / self.disag_models), axis=0)

        # log_prob_state_mask1 = tf.reduce_logsumexp(
        #     tf.cast(log_prob_state_mask1, tf.float32) + tf.math.log(1 / num_samples), axis=0)

        reward = -tf.reduce_mean(log_prob_state_mask, -1) / 10

        # self.Normalizer.update(
        #     tf.stop_gradient(tf.cast(tf.reshape(reward, [-1, 1]), 'float32')))
        # reward = self.Normalizer(reward)
        # reward = 1 * tf.reduce_mean(
        #     tf.cast(log_prob_state, tf.float32) - log_prob2, axis=-1)

        # if self._config.expl_extr_scale:
        #     reward += tf.cast(self._config.expl_extr_scale * self._reward(
        #         feat, state, action), tf.float32)
        return reward
