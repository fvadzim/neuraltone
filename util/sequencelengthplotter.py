'''
This python program reads in all the source text data, determining the length of each
file and plots them in a histogram.

This is used to help determine the bucket sizes to be used in the main program.
'''

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import nltk
import sys
import numpy as np
import os
from util import twittertokenizer

# dirs = ["data/aclImdb/test/pos", "data/aclImdb/test/neg", "data/aclImdb/train/pos", "data/aclImdb/train/neg"]
# dirs = ["../data/twitter/test/pos", "../data/twitter/test/neg",
#         "../data/twitter/train/pos", "../data/twitter/train/neg"]
dirs = ["../data/twitter/test/pos"]
num_bins = 25
string_args = [("num_bins", "int")]


def main():
    lengths = []
    count = 0
    tokenizer = twittertokenizer.TweetTokenizer(preserve_case=False)
    for d in dirs:
        print("Grabbing sequence lengths from: {0}".format(d))
        for f in os.listdir(d):
            count += 1
            if count % 1000 == 0:
                print("Determining length of: {0}".format(f))
                break
            with open(os.path.join(d, f), 'r', encoding='UTF-8') as review:
                tokens = tokenizer.tokenize(review.read())
                numTokens = len(tokens)
                lengths.append(numTokens)

    # mu = np.mean(lengths)
    # sigma = np.std(lengths)
    mu = np.std(lengths)
    sigma = np.mean(lengths)
    x = np.array(lengths)
    n, bins, patches = plt.hist(x, num_bins, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.title("Frequency of Sequence Lengths")
    plt.xlabel("Length")
    plt.ylabel("Number of Sequences")
    plt.xlim(0, 100)
    plt.show()


'''
Command line arguments being read in
'''


def setGraphParameters():
    try:
        for arg in string_args:
            exec("if \"{0}\" in sys.argv:\n\
                \tglobal {0}\n\
                \t{0} = {1}(sys.argv[sys.argv.index({0}) + 1])".format(arg[0], arg[1]))
    except Exception as a:
        print("Problem with cmd args " + str(a))


'''
This function tokenizes sentences
'''


if __name__ == "__main__":
    main()
