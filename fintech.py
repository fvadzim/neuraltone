"""
http://deeplearning.net/tutorial/lstm.html

Adapted by: Dmitry Fisko @dmitryfisko
"""

import configparser

import numpy as np
import tensorflow as tf

import models.sentiment
import util.dataprocessor
import util.vocabmapping
from models_db import *
from util import twittertokenizer


class ToneRus(object):
    new_token = {
        "коррупц": "плохо",
        "выгод": "хорошо",
        "финанс": "процент",
        "карточк": "карта",
        "технология": "процент",
        "уголовн": "уголовн",
        "ценн": "хорошо",
        "наруш": "наруш",
        "отлично": "хорошо",
        "поддерж": "процент",
        "польза": "хорошо"
    }

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('checkpoint_dir', 'data/checkpoints/',
                        'Directory to store/restore checkpoints')
    flags.DEFINE_string('text', "Хорош", 'Text to sample with.')
    flags.DEFINE_string('config_file', 'config.ini', 'Path to configuration file.')

    def __init__(self):
        self.vocab_mapping = util.vocabmapping.VocabMapping()
        self.tokenizer = twittertokenizer.TweetTokenizer(preserve_case=False)
        self.sess = tf.Session()
        self.model = self.load_model(self.sess, self.vocab_mapping.getSize())
        if self.model is None:
            return
        self.max_seq_length = self.model.max_seq_length
        # test_data = [FLAGS.text.lower()]

    def __del__(self):
        self.sess.close()

    def prepareText(self, tokenizer, text, max_seq_length, vocab_mapping):
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
            for new_token in self.new_token.keys():
                if new_token in tokens[index]:
                    tokens[index] = self.new_token.get(new_token, tokens[index])

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

    def get_positive(self, text):
        text = text.lower()
        data, seq_lengths, targets = self.prepareText(self.tokenizer, text, self.max_seq_length,
                                                      self.vocab_mapping)
        input_feed = {self.model.seq_input.name: data, self.model.target.name: targets,
                      self.model.seq_lengths.name: seq_lengths}
        output_feed = [self.model.y]
        outputs = self.sess.run(output_feed, input_feed)
        # score = np.argmax(outputs[0])
        # probability = outputs[0].max(axis=1)[0]

        # print(str(outputs))
        # print(text)
        # print("Value of sentiment: negative probability: {0}".format(outputs[0][0][0]))
        # print("Value of sentiment: positive probability: {0}".format(outputs[0][0][1]))

        return outputs[0][0][1]

    def load_model(self, session, vocab_size):
        hyper_params = self.read_config_file()
        model = models.sentiment.SentimentModel(vocab_size,
                                                hyper_params["hidden_size"],
                                                1.0,
                                                hyper_params["num_layers"],
                                                hyper_params["grad_clip"],
                                                hyper_params["max_seq_length"],
                                                hyper_params["learning_rate"],
                                                hyper_params["lr_decay_factor"],
                                                1)
        ckpt = tf.train.get_checkpoint_state(self.FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Reading model parameters from {0}".format(ckpt.model_checkpoint_path))
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Double check you got the checkpoint_dir right...")
            print("Model not found...")
            model = None
        return model

    def read_config_file(self):
        """
        Reads in config file, returns dictionary of network params
        """
        config = configparser.ConfigParser()
        config.read(self.FLAGS.config_file)
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


def main():
    # load trained model
    tone = ToneRus()

    # main function - tone.get_positive
    while True:
        print(tone.get_positive(input('Enter line: ').strip()))


def generator():
    arr = set()
    for index, item in enumerate(Tonalities.select()):
        if item in arr:
            continue
        arr.add(item.article)
        try:
            publication = Publications.get(article=item.article)
        except:
            continue
        if not publication.content:
            continue
        yield publication

if __name__ == "__main__":
    # see main
    main()

    #
    # for pub in generator():
    #     print(pub.url)

