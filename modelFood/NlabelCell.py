
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

from .config import Config


_Linear = core_rnn_cell._Linear


class NLabelCellWrapper(rnn_cell_impl.RNNCell):


  def __init__(self, cell, state_is_tuple=True, reuse=None,emb_M3=None,emb_M4k=None):

    super(NLabelCellWrapper, self).__init__(_reuse=reuse)

    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))

    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)

    self._state_is_tuple = state_is_tuple
    self._cell = cell

    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

    self.emb_M3 =emb_M3
    self.emb_M4k = emb_M4k

    self.config = Config()



  @property
  def state_size(self):
    size = (self._cell.state_size, self.config.use_K_histroy*self.config.label_emb_size)
    if self._state_is_tuple:
            #print("state return size",size)
            return size
    else:
            return sum(list(size))

  @property
  def output_size(self):
    return self.config.label_emb_size

  def call(self, inputs, state):
      """Long short-term memory cell with attention (LSTMA)."""
      if self._state_is_tuple:
        state, hisTrack= state

      #print("inputs",inputs)

      #
      # print("state",state)

      cell_output, new_state = self._cell(inputs,state)
      #print("cell_out",cell_output)
      #print("new_state",new_state)

      c_prev,m_prev = new_state

      m_prev=tf.expand_dims(m_prev,axis=1)

      #print("c_prev",c_prev)
      #print("m_prev", m_prev)


      label_emb= tf.nn.relu(tf.matmul(cell_output,self.emb_M3))
      label_emb = tf.expand_dims(label_emb,axis=1)
      #print(label_emb)
      hisTrack=tf.reshape(hisTrack,shape=[-1,self.config.use_K_histroy,self.config.label_emb_size])

      new_hisTrack = tf.slice(hisTrack, [0,1, 0], [-1,self.config.use_K_histroy-1, self.config.label_emb_size])

      #print("new_hisTrack", new_hisTrack)
      #print("label_emb", label_emb)

      concat_hisTrack=tf.concat([new_hisTrack,label_emb],axis=1)
      #print("concat_hisTrack_tmp",concat_hisTrack)

      concat_all= tf.concat([concat_hisTrack,m_prev],axis=1)

      concat_all_flatten=tf.reshape(concat_all,shape=[-1,(self.config.use_K_histroy+1)*self.config.label_emb_size])

      concat_hisTrack_flatten=tf.reshape(concat_hisTrack,shape=[-1,self.config.use_K_histroy*self.config.label_emb_size])

      m =tf.nn.relu(tf.matmul(concat_all_flatten,self.emb_M4k))

      new_state_tuple= LSTMStateTuple(cell_output,m)

      new_send_state=(new_state_tuple,concat_hisTrack_flatten)

      #print("new_send_state",new_send_state)
      #print("cell_output",cell_output)

      return cell_output, new_send_state




class NLabelCellWrapperAfterAtt(rnn_cell_impl.RNNCell):


  def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=True, reuse=None,emb_M3=None,emb_M4k=None):

    super(NLabelCellWrapperAfterAtt, self).__init__(_reuse=reuse)

    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError("attn_length should be greater than zero, got %s"
                       % str(attn_length))
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

    self.emb_M3 =emb_M3
    self.emb_M4k = emb_M4k

    self.config = Config()



  @property
  def state_size(self):
    size = (self._cell.state_size, self.config.use_K_histroy*self.config.label_emb_size)
    if self._state_is_tuple:
            #print("state return size",size)
            return size
    else:
            return sum(list(size))

  @property
  def output_size(self):
    return self.config.label_emb_size

  def call(self, inputs, state):
      """Long short-term memory cell with attention (LSTMA)."""
      if self._state_is_tuple:
        state, hisTrack= state
        states ,attns, attn_states=state

      #print("inputs",inputs)

      #print("state",state)

      cell_output, new_state = self._cell(inputs,state)
      #print("cell_out",cell_output)
      #print("new_state",new_state)

      new_state, _ ,_ =new_state

      c_prev,m_prev = new_state

      m_prev=tf.expand_dims(m_prev,axis=1)

      #print("c_prev",c_prev)
      #print("m_prev", m_prev)


      label_emb= tf.nn.relu(tf.matmul(cell_output,self.emb_M3))
      label_emb = tf.expand_dims(label_emb,axis=1)
      #print(label_emb)
      hisTrack=tf.reshape(hisTrack,shape=[-1,self.config.use_K_histroy,self.config.label_emb_size])

      new_hisTrack = tf.slice(hisTrack, [0,1, 0], [-1,self.config.use_K_histroy-1, self.config.label_emb_size])

      #print("new_hisTrack", new_hisTrack)
      #print("label_emb", label_emb)

      concat_hisTrack=tf.concat([new_hisTrack,label_emb],axis=1)
      #print("concat_hisTrack_tmp",concat_hisTrack)

      concat_all= tf.concat([concat_hisTrack,m_prev],axis=1)

      concat_all_flatten=tf.reshape(concat_all,shape=[-1,(self.config.use_K_histroy+1)*self.config.label_emb_size])

      concat_hisTrack_flatten=tf.reshape(concat_hisTrack,shape=[-1,self.config.use_K_histroy*self.config.label_emb_size])

      m =tf.nn.relu(tf.matmul(concat_all_flatten,self.emb_M4k))

      new_state_tuple= (LSTMStateTuple(cell_output,m),attns,attn_states)

      new_send_state=(new_state_tuple,concat_hisTrack_flatten)

      #print("new_send_state",new_send_state)
      #print("cell_output",cell_output)

      return cell_output, new_send_state


class LabelCellWrapperAfterAtt(rnn_cell_impl.RNNCell):


  def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=True, reuse=None,emb_M3=None,emb_M4=None):

    super(LabelCellWrapperAfterAtt, self).__init__(_reuse=reuse)
    # if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
    #   raise TypeError("The parameter cell is not RNNCell.")
    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError("attn_length should be greater than zero, got %s"
                       % str(attn_length))
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell

    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

    self.emb_M3 =emb_M3
    self.emb_M4 = emb_M4
    self.config = Config()


  @property
  def state_size(self):
    size = (self._cell.state_size)
    if self._state_is_tuple:
            #print("state return size",size)
            return size
    else:
            return sum(list(size))

  @property
  def output_size(self):
    return self.config.label_emb_size


  def call(self, inputs, state):
      """Long short-term memory cell with attention (LSTMA)."""
      if self._state_is_tuple:
        states ,attns, attn_states=state

      #print("inputs",inputs)

      #print("state",state)

      cell_output, new_state = self._cell(inputs,state)

      #print("cell_out",cell_output)
      #print("new_state",new_state)

      new_state, _ ,_ =new_state

      c_prev,m_prev = new_state

      m_prev=tf.expand_dims(m_prev,axis=1)

      #print("c_prev",c_prev)
      #print("m_prev", m_prev)


      label_emb= tf.nn.relu(tf.matmul(cell_output,self.emb_M3))
      label_emb = tf.expand_dims(label_emb,axis=1)
      #print(label_emb)


      concat_all= tf.concat([label_emb,m_prev],axis=1)

      concat_all_flatten=tf.reshape(concat_all,shape=[-1,2*self.config.label_emb_size])


      m =tf.nn.relu(tf.matmul(concat_all_flatten,self.emb_M4))

      new_state_tuple= (LSTMStateTuple(cell_output,m),attns,attn_states)

      new_send_state=new_state_tuple

      #print("new_send_state",new_send_state)
      #print("cell_output",cell_output)

      return cell_output, new_send_state


class LabelCellWrapper(rnn_cell_impl.RNNCell):

  def __init__(self, cell, state_is_tuple=True, reuse=None,emb_M3=None,emb_M4=None):

    super(LabelCellWrapper, self).__init__(_reuse=reuse)

    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))

    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)

    self._state_is_tuple = state_is_tuple
    self._cell = cell

    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

    self.emb_M3 =emb_M3
    self.emb_M4 = emb_M4

    self.config = Config()



  @property
  def state_size(self):
    size = (self._cell.state_size)
    if self._state_is_tuple:
            #print("state return size",size)
            return size
    else:
            return sum(list(size))

  @property
  def output_size(self):
    return self.config.label_emb_size


  #before attention
  def call(self, inputs, state):

      if self._state_is_tuple:
        state= state

      #print("inputs",inputs)

      #print("state",state)

      cell_output, new_state = self._cell(inputs, state)
      #print("cell_out",cell_output)
      #print("new_state",new_state)

      c_prev,m_prev = new_state

      m_prev=tf.expand_dims(m_prev,axis=1)

      #print("c_prev",c_prev)
      #print("m_prev", m_prev)


      label_emb= tf.nn.relu(tf.matmul(cell_output,self.emb_M3))
      label_emb = tf.expand_dims(label_emb,axis=1)
      #print(label_emb)


      concat_all= tf.concat([label_emb,m_prev],axis=1)

      concat_all_flatten=tf.reshape(concat_all,shape=[-1,2*self.config.label_emb_size])


      m =tf.nn.relu(tf.matmul(concat_all_flatten,self.emb_M4))

      new_state_tuple= LSTMStateTuple(cell_output,m)

      new_send_state=new_state_tuple

      #print("new_send_state",new_send_state)
      #print("cell_output",cell_output)

      return cell_output, new_send_state


class NLabelAttentionCellWrapper(rnn_cell_impl.RNNCell):


  def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=True, reuse=None,emb_M3=None,emb_M4k=None):

    super(NLabelAttentionCellWrapper, self).__init__(_reuse=reuse)
    #print("build attentionCellWrapper")
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("The parameter cell is not RNNCell.")
    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError("attn_length should be greater than zero, got %s"
                       % str(attn_length))
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

    self.emb_M3 = emb_M3
    self.emb_M4k = emb_M4k
    self.config = Config()




  @property
  def state_size(self):
    size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length,self.config.use_K_histroy*self.config.label_emb_size)
    if self._state_is_tuple:
      return size
    else:
      return sum(list(size))

  @property
  def output_size(self):
    return self._attn_size

  def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""
    if self._state_is_tuple:
      state, attns, attn_states,histotry = state

    else:
      states = state
      state = array_ops.slice(states, [0, 0], [-1, self._cell.state_size])
      attns = array_ops.slice(
          states, [0, self._cell.state_size], [-1, self._attn_size])
      attn_states = array_ops.slice(
          states, [0, self._cell.state_size + self._attn_size],
          [-1, self._attn_size * self._attn_length])


    attn_states = array_ops.reshape(attn_states,
                                    [-1, self._attn_length, self._attn_size])
    input_size = self._input_size
    if input_size is None:
      input_size = inputs.get_shape().as_list()[1]
    if self._linear1 is None:
      self._linear1 = _Linear([inputs, attns], input_size, True)
    inputs = self._linear1([inputs, attns])

    cell_output, new_state = self._cell(inputs, state)
    #print("new state",new_state)


    if self._state_is_tuple:
      new_state_cat = array_ops.concat(nest.flatten(new_state), 1)
    else:
      new_state_cat = new_state
    new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
    with vs.variable_scope("attn_output_projection"):
      if self._linear2 is None:
        self._linear2 = _Linear([cell_output, new_attns], self._attn_size, True)

      output = self._linear2([cell_output, new_attns])

    #print("output",output)

    new_attn_states = array_ops.concat(
        [new_attn_states, array_ops.expand_dims(output, 1)], 1)
    new_attn_states = array_ops.reshape(
        new_attn_states, [-1, self._attn_length * self._attn_size])

    c_new, h_new = new_state

    #print("c_new", c_new)
    #print("h_new", h_new)

    label_emb = tf.nn.relu(tf.matmul(output, self.emb_M3))
    # label_emb = tf.expand_dims(label_emb, axis=1)
    #print("label emb",label_emb)
    #print("new stat", new_state)

    pre_history = histotry
    pre_history= tf.reshape(pre_history, shape=[-1, self.config.use_K_histroy, self.config.label_emb_size])
    #print("pre_history",pre_history)

    new_history = tf.slice(pre_history, [0, 1, 0], [-1, self.config.use_K_histroy - 1, self.config.label_emb_size])

    #print("new_history", new_history)
    # print("label_emb", label_emb)

    concat_his = tf.concat([new_history, tf.expand_dims(label_emb,axis=1)], axis=1)
    #print("concat_his_tmp", concat_his)

    concat_all = tf.concat([concat_his,  tf.expand_dims(c_new,axis=1)], axis=1)

    #print("c_new",c_new)

    concat_all_flatten = tf.reshape(concat_all,
                                shape=[-1, (self.config.use_K_histroy + 1) * self.config.label_emb_size])

    concat_his_flatten = tf.reshape(concat_his,
                                     shape=[-1, self.config.use_K_histroy * self.config.label_emb_size])

    c = tf.nn.relu(tf.matmul(concat_all_flatten, self.emb_M4k))

    new_state= LSTMStateTuple(c, h_new)

    new_wrapper_state = (new_state, new_attns, new_attn_states, concat_his_flatten)



    return output, new_wrapper_state

  def _attention(self, query, attn_states):
    conv2d = nn_ops.conv2d
    reduce_sum = math_ops.reduce_sum
    softmax = nn_ops.softmax
    tanh = math_ops.tanh

    with vs.variable_scope("attention"):
      k = vs.get_variable(
          "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
      v = vs.get_variable("attn_v", [self._attn_vec_size])
      hidden = array_ops.reshape(attn_states,
                                 [-1, self._attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      if self._linear3 is None:
        self._linear3 = _Linear(query, self._attn_vec_size, True)
      y = self._linear3(query)
      y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = array_ops.reshape(d, [-1, self._attn_size])
      new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
      return new_attns, new_attn_states


class NLabelNoAttentionCellWrapper(rnn_cell_impl.RNNCell):

  def __init__(self, cell,
               input_size=None, state_is_tuple=True, reuse=None,emb_M3=None,emb_M4k=None):

    super(NLabelNoAttentionCellWrapper, self).__init__(_reuse=reuse)
  #
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
      raise TypeError("The parameter cell is not RNNCell.")
    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))

    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)

    self._state_is_tuple = state_is_tuple
    self._cell = cell

    self._input_size = input_size


    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

    self.emb_M3 = emb_M3
    self.emb_M4k = emb_M4k
    self.config = Config()
    self._output_size=  cell.output_size




  @property
  def state_size(self):
    size = (self._cell.state_size,self.config.use_K_histroy*self.config.label_emb_size)
    if self._state_is_tuple:
      return size
    else:
      return sum(list(size))

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""
    if self._state_is_tuple:
      state,histotry = state

    cell_output, new_state = self._cell(inputs, state)
    #print("new state",new_state)



    output = cell_output

    #print("output",output)


    c_new, h_new = new_state


    # print("c_new", c_new)
    # print("h_new", h_new)

    label_emb = tf.nn.relu(tf.matmul(output, self.emb_M3))
    # label_emb = tf.expand_dims(label_emb, axis=1)
    #print("label emb",label_emb)
    #print("new stat", new_state)

    pre_history = histotry
    pre_history= tf.reshape(pre_history, shape=[-1, self.config.use_K_histroy, self.config.label_emb_size])
    #print("pre_history",pre_history)

    new_history = tf.slice(pre_history, [0, 1, 0], [-1, self.config.use_K_histroy - 1, self.config.label_emb_size])

    #print("new_history", new_history)
    # print("label_emb", label_emb)

    concat_his = tf.concat([new_history, tf.expand_dims(label_emb,axis=1)], axis=1)
    #print("concat_his_tmp", concat_his)

    concat_all = tf.concat([concat_his,  tf.expand_dims(c_new,axis=1)], axis=1)

    #print("c_new",c_new)

    concat_all_flatten = tf.reshape(concat_all,
                                shape=[-1, (self.config.use_K_histroy + 1) * self.config.label_emb_size])

    concat_his_flatten = tf.reshape(concat_his,
                                     shape=[-1, self.config.use_K_histroy * self.config.label_emb_size])

    c = tf.nn.relu(tf.matmul(concat_all_flatten, self.emb_M4k))

    new_state= LSTMStateTuple(c, h_new)

    new_wrapper_state = (new_state, concat_his_flatten)



    return output, new_wrapper_state


