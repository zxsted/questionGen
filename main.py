import tensorflow as tf
import random
from nltk.translate.bleu_score import corpus_bleu

from model import QuestionGen

tf.app.flags.DEFINE_integer("batch_size", 128, "Num of batch.")
tf.app.flags.DEFINE_integer("embed_units", 128, "Size of word embed.")
tf.app.flags.DEFINE_integer("num_units", 128, "Size of hidden state.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers.")
tf.app.flags.DEFINE_integer("max_len", 10, "Max sequence length.")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "Size of vocabulary.")
tf.app.flags.DEFINE_integer("num_epoch", 20, "Number of epoch.")
tf.app.flags.DEFINE_integer("per_step", 100, "Print per steps.")
tf.app.flags.DEFINE_string("data_dir", "data", "Path of data dir.")
tf.app.flags.DEFINE_string("train_dir", "/home/data/zhangzheng/model/questionGen", "Training dir.")
tf.app.flags.DEFINE_string("vocab_path", "vocab.txt", "Path of vocab file.")
tf.app.flags.DEFINE_integer("inference_version", 0, "The version for inference.")
tf.app.flags.DEFINE_boolean("is_train", True, "If is training.")
FLAGS = tf.app.flags.FLAGS


def train(sess, model, batch_data):
    a = 0


def inference(sess, model, test_data):
    a = 0


def load_vocab(vocab_dir):
    with open(vocab_dir) as f:
        words = [line.strip().split()[0] for line in f.readlines()]
    assert(len(words) > FLAGS.vocab_size)
    words = words[:FLAGS.vocab_size]
    vocab = dict(zip(words, range(len(words))))
    return vocab, words


def load_data(data_path):
    with open(data_path) as f:
        data = [line.strip().split() for line in f.readlines()]
    return data


def combine_data(paragraphs, questions, answers, answer_sents):
    data = list()
    assert(len(paragraphs) == len(questions) == len(answers) == len(answer_sents))
    for i in range(len(paragraphs)):
        _dict = {'paragraph': paragraphs[i], 'question': questions[i],
                 'answer': answers[i], 'answer_sent': answer_sents[i]}
        data.append(_dict)
    return data


def split_data(data, train_ratio, dev_ratio):
    size1 = int(len(data) * train_ratio)
    size2 = int(len(data) * dev_ratio)
    return data[:size1], data[size1:size1+size2], data[size1+size2:]


def get_batched_data(data):
    selected_data = [random.choice(data) for _ in range(FLAGS.batch_size)]
    return convert_data(selected_data)


def convert_data(batched_data, vocab, max_len=-1):
    """
    Args:
        batched_data: list of dicts, just like load_data() return
        vocab: dict, word2index
        max_len: dict, data_source2max_len
    Returns:
        new_data: list of dicts, like batched_data
    """
    paragraphs = [item['answer'] for item in batched_data]
    answers = [item['answer'] for item in batched_data]
    questions = [item['question'] for item in batched_data]
    answer_sents = [item['answer_sent'] for item in batched_data]

    def _convert_data(_data, vocab, max_len):
        if max_len == -1:
            max_len = max([len(item) for item in _data]) + 1
        lens, new_data = [], []
        for item in _data:
            lens.append(len(item) + 1)
            new_item = []
            for word in item:
                if word in vocab:
                    new_item.append(vocab[word])
                else:
                    new_item.append(1)
            new_item += [3] + [1] * (max_len - len(new_item) - 1)
            new_data.append(new_item)
        return new_data, lens

    answers, answer_lens = convert_data(answers, vocab, max_len)
    questions, question_lens = convert_data(questions, vocab, max_len)
    answer_sents, answer_sent_lens = convert_data(answer_sents, vocab, max_len)

    return {'answer': answers, 'question': questions, 'answer_sent': answer_sents,
            'answer_len': answer_lens, 'question_len': question_lens,
            'answer_sent_len': answer_sent_lens}


print("loading vocab...")
vocab, words = load_vocab("{}/{}".format(FLAGS.data_dir, FLAGS.vocab_path))
answers = load_data("{}/{}.txt".format(FLAGS.data_dir, 'answer'))
questions = load_data("{}/{}.txt".format(FLAGS.data_dir, 'question'))
paragraphs = load_data("{}/{}.txt".format(FLAGS.data_dir, 'paragraph'))
answer_sents = load_data("{}/{}.txt".format(FLAGS.data_dir, 'answer_sent'))

data = combine_data(paragraphs, questions, answers, answer_sents)
train_data, dev_data, test_data = split_data(data, 0.95, 0.02)
print("train_size: {}\ndev_size: {}\ntest_size: {}".format(len(train_data), len(dev_data), len(test_data)))

_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True
print("start session...")
with tf.Session(config=_config) as sess:
    if FLAGS.is_train:
        print("TRAINING")
        model = QuestionGen(FLAGS.num_units,
                            FLAGS.embed_units,
                            FLAGS.num_layers,
                            FLAGS.vocab_size)
        if tf.train.get_checkpoint_state(FLAGS.train_dir):
            print("Restore parameters from %s..." % FLAGS.train_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
        else:
            print("Create model with fresh parameters...")
            sess.run(tf.global_variables_initializer())

        while True:
            global_step = model.global_step.eval()
            if global_step > 0 and global_step % FLAGS.per_step == 0:
                print("step: {}".format(global_step))
                model.saver.save(sess, "{}/checkpoint".format(FLAGS.train_dir),
                                 global_step=model.global_step)
            loss = train(sess, model, get_batched_data(data))
    else:
        a = 0
