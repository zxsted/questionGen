import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell

class QuestionGen:

    def __init__(self,
                 num_units,
                 embed_units,
                 num_layers,
                 vocab_size,
                 is_training = True,
                 ):
        self.global_step = tf.Variable(0, trainable=False)
        self.embedding = tf.get_variable("embedding", [vocab_size, embed_units], tf.float32)

        self.answer_sent = tf.placeholder(tf.int32, [None, None], "input_answer_sent")
        self.answer_sent_len = tf.placeholder(tf.int32, [None], "input_answer_sent_length")
        self.answer = tf.placeholder(tf.int32, [None, None], "input_answer")
        self.answer_len = tf.placeholder(tf.int32, [None], "input_answer_length")
        self.question = tf.placeholder(tf.int32, [None, None], "question")
        self.question_len = tf.placeholder(tf.int32, [None], "question_length")
        decoder_input = tf.concat((tf.ones_like(self.question[:, :1])*2, self.y[:, :-1]), -1)  # 2: _GO

        cell_as = MultiRNNCell([GRUCell(num_units)] * num_layers)
        cell_a = MultiRNNCell([GRUCell(num_units)] * num_layers)
        answer_sent_ = tf.nn.embedding_lookup(self.embedding, self.answer_sent)  # [batch * len * embed_size]
        answer_ = tf.nn.embedding_lookup(self.embedding, self.answer)
        decoder_input_ = tf.nn.embedding_lookup(self.embedding, decoder_input)

        hc = bidirectional_dynamic_rnn(cell_as, answer_sent_, self.answer_sent_len)
        ha = dynamic_rnn(cell_a, answer_, self.answer_len)
