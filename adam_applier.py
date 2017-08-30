# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
class AdamApplier(object):

  def __init__(self,
               learning_rate,
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               clip_norm=40.0,
               name="Adam",
               ):

    self._name = name
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._clip_norm = clip_norm
    self._epsilon = epsilon
    # Tensor versions of the constructor arguments, created in _prepare()
    self._learning_rate_tensor = None
    self._beta1_tensor = None
    self._beta2_tensor = None
    self._epsilon_tensor = None

    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None

    self._slots = {}

  def _get_beta_accumulators(self):
        return self._beta1_power, self._beta2_power

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable.
    if (self._beta1_power is None or
        self._beta1_power.graph is not var_list[0].graph):
      with ops.colocate_with(var_list[0]):
        self._beta1_power = variables.Variable(self._beta1,
                                               name="beta1_power",
                                               trainable=False)
        self._beta2_power = variables.Variable(self._beta2,
                                               name="beta2_power",
                                               trainable=False)
    for v in var_list:
      # 'val' is Variable's intial value tensor.
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
      self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate,
                                                      name="learning_rate")
      self._beta1_tensor = tf.convert_to_tensor(self._beta1, name="decay")
      self._beta2_tensor = tf.convert_to_tensor(self._beta2,
                                                 name="momentum")
      self._epsilon_tensor = tf.convert_to_tensor(self._epsilon,
                                                name="epsilon")

  def _slot_dict(self, slot_name):
    named_slots = self._slots.get(slot_name, None)
    if named_slots is None:
      named_slots = {}
      self._slots[slot_name] = named_slots
    return named_slots

  def _get_or_make_slot(self, var, val, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_slot(var, val, op_name)
    return named_slots[var]

  def get_slot(self, var, name):
    named_slots = self._slots.get(name, None)
    if not named_slots:
      return None
    return named_slots.get(var, None)

  def _zeros_slot(self, var, slot_name, op_name):
    named_slots = self._slot_dict(slot_name)
    if var not in named_slots:
      named_slots[var] = slot_creator.create_zeros_slot(var, op_name)
    return named_slots[var]

  # TODO: in RMSProp native code, memcpy() (for CPU) and
  # cudaMemcpyAsync() (for GPU) are used when updating values,
  # and values might tend to be overwritten with results from other threads.
  # (Need to check the learning performance with replacing it)
  def _apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.apply_adam(
      var, m, v,
      self._beta1_power,
      self._beta2_power,
      self._learning_rate_tensor,
      self._beta1_tensor,
      self._beta2_tensor,
      self._epsilon_tensor,
      grad,
      use_locking=False).op

  # Apply accumulated gradients to var.
  def apply_gradients(self, var_list, accum_grad_list, name=None):
    update_ops = []


    with tf.control_dependencies(None):
        self._create_slots(var_list)

    with tf.name_scope(name, self._name, []) as name:
        self._prepare()
        for var, accum_grad in zip(var_list, accum_grad_list):
          with tf.name_scope("update_" + var.op.name), tf.device(var.device):
            clipped_accum_grad = tf.clip_by_norm(accum_grad, self._clip_norm)
            update_ops.append(self._apply_dense(clipped_accum_grad, var))
        return tf.group(*update_ops, name=name)
