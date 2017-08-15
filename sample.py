"""
http://deeplearning.net/tutorial/lstm.html

Adapted by: Dmitry Fisko @dmitryfisko
"""

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import nltk
import util.dataprocessor
import models.sentiment
import util.vocabmapping
import configparser
from util import twittertokenizer

new_token = {
    "коррупция": "плохо"
}

test_data = (
    # 'Буду ли я помогать Алексею Навальному в его деле? На сегодняшний момент — нет. Здесь я солидарен с Максимом Кацем',
    # 'Хорошо чувствовать себя — собой: ничьим, непонятным самому себе, уютным и домашним, шестилетним, вечным. Хорошо любить и не ждать подвоха',
    # 'Люблю спать, но спать не любит меня',
    # 'Дело по ложному доносу в отношении заключенного не будет возбуждаться, так как он не составлял заявление лично:',
    # 'отличный день',
    # 'как твой день проходил. ну как тебе сказать...',
    "в банке нет коррупции",
)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/', 'Directory to store/restore checkpoints')
flags.DEFINE_string('text', "Хорош", 'Text to sample with.')
flags.DEFINE_string('config_file', 'config.ini', 'Path to configuration file.')


def main():

    with open('data/obscene_corpus.txt', 'r') as f:
        for mat in f:	
            new_token[mat.lower()] = 'плохой'

    vocab_mapping = util.vocabmapping.VocabMapping()
    tokenizer = twittertokenizer.TweetTokenizer(preserve_case=False)
    with tf.Session() as sess:
        model = load_model(sess, vocab_mapping.getSize())
        if model == None:
            return
        max_seq_length = model.max_seq_length
        # test_data = [FLAGS.text.lower()]

        while True:
            text = input("Enter phrase: ").lower()
            data, seq_lengths, targets = prepareText(tokenizer, text, max_seq_length, vocab_mapping)
            input_feed = {model.seq_input.name: data, model.target.name: targets,
                          model.seq_lengths.name: seq_lengths}
            output_feed = [model.y]
            outputs = sess.run(output_feed, input_feed)
            # score = np.argmax(outputs[0])
            # probability = outputs[0].max(axis=1)[0]
            print(str(outputs))
            print(text)
            print("Value of sentiment: negative probability: {0}".format(outputs[0][0][0]))
            print("Value of sentiment: positive probability: {0}".format(outputs[0][0][1]))


        # for text in test_data:
        #     text = text.lower()
        #     data, seq_lengths, targets = prepareText(tokenizer, text, max_seq_length, vocab_mapping)
        #     input_feed = {model.seq_input.name: data, model.target.name: targets,
        #                   model.seq_lengths.name: seq_lengths}
        #     output_feed = [model.y]
        #     outputs = sess.run(output_feed, input_feed)
        #     # score = np.argmax(outputs[0])
        #     # probability = outputs[0].max(axis=1)[0]
        #     print(str(outputs))
        #     print(text)
        #     print("Value of sentiment: negative probability: {0}".format(outputs[0][0][0]))
        #     print("Value of sentiment: positive probability: {0}".format(outputs[0][0][1]))


def prepareText(tokenizer, text, max_seq_length, vocab_mapping):
    """
    Input:
    text_list: a list of strings

    Returns:
    inputs, seq_lengths, targets
    """
    data = np.array([i for i in range(max_seq_length)])
    targets = []
    seq_lengths = []
    tokens = tokenizer.tokenize(text)

    for index, token in enumerate(tokens):
        tokens[index] = new_token.get(token, token)

    if len(tokens) > max_seq_length:
        tokens = tokens[0:max_seq_length]
    inputs = []

    indices = [vocab_mapping.getIndex(j) for j in tokens]
    if len(indices) < max_seq_length:
        indices = indices + [vocab_mapping.getIndex("<PAD>") for i in
                             range(max_seq_length - len(indices))]
    else:
        indices = indices[0:max_seq_length]
    seq_lengths.append(len(tokens))

    data = np.vstack((data, indices))
    targets.append(1)

    onehot = np.zeros((len(targets), 2))
    onehot[np.arange(len(targets)), targets] = 1
    return data[1::], np.array(seq_lengths), onehot


def load_model(session, vocab_size):
    hyper_params = read_config_file()
    model = models.sentiment.SentimentModel(vocab_size,
                                            hyper_params["hidden_size"],
                                            1.0,
                                            hyper_params["num_layers"],
                                            hyper_params["grad_clip"],
                                            hyper_params["max_seq_length"],
                                            hyper_params["learning_rate"],
                                            hyper_params["lr_decay_factor"],
                                            1)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Double check you got the checkpoint_dir right...")
        print("Model not found...")
        model = None
    return model


def read_config_file():
    """
    Reads in config file, returns dictionary of network params
    """
    config = configparser.ConfigParser()
    config.read(FLAGS.config_file)
    dic = {}
    sentiment_section = "sentiment_network_params"
    general_section = "general"
    dic["num_layers"] = config.getint(sentiment_section, "num_layers")
    dic["hidden_size"] = config.getint(sentiment_section, "hidden_size")
    dic["dropout"] = config.getfloat(sentiment_section, "dropout")
    dic["batch_size"] = config.getint(sentiment_section, "batch_size")
    dic["train_frac"] = config.getfloat(sentiment_section, "train_frac")
    dic["learning_rate"] = config.getfloat(sentiment_section, "learning_rate")
    dic["lr_decay_factor"] = config.getfloat(sentiment_section, "lr_decay_factor")
    dic["grad_clip"] = config.getint(sentiment_section, "grad_clip")
    dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
                                                                    "use_config_file_if_checkpoint_exists")
    dic["max_epoch"] = config.getint(sentiment_section, "max_epoch")
    dic["max_vocab_size"] = config.getint(sentiment_section, "max_vocab_size")
    dic["max_seq_length"] = config.getint(general_section,
                                          "max_seq_length")
    dic["steps_per_checkpoint"] = config.getint(general_section,
                                                "steps_per_checkpoint")
    return dic


if __name__ == "__main__":
    main()
