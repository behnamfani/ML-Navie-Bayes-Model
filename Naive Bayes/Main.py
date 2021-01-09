from __future__ import division
import tokenizer
import math
import numpy as np
import os
import scipy.interpolate as sc
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = "D:/Python/CODE/NaiveBayes/large_movie_review_dataset"
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")


def tokenize_doc(doc):
    """
    To tokenize our doc, first, I make it lower. Then there are 2 arrays to keep track of the words and the number of
    their appearances in the doc. So with a for loop over the tokens, if the token was a word, I update these arrays.
    In the end, I make a dictionary using these arrays.
    """
    word, number = [], []
    doc = doc.lower()
    for token in tokenizer.tokenize(doc):
        kind, txt, val = token
        if kind == tokenizer.TOK.WORD:
            if txt in word:
                for i in range(len(word)):
                    if word[i] == txt:
                        number[i] += 1
            else:
                word.append(txt)
                number.append(1)
            pass
        else:
            pass
    f = {}
    for i in range(len(word)):
        f[word[i]] = number[i]

    return f


class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the training set of that class
        self.class_total_doc_counts = {POS_LABEL: 0.0,
                                       NEG_LABEL: 0.0}

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = {POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0}

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = {POS_LABEL: defaultdict(float),
                                  NEG_LABEL: defaultdict(float)}

    def train_model(self, num_docs=None):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print("Limiting to only %s docs per class" % num_docs)

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print("Starting training with paths %s and %s" % (pos_path, neg_path))
        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p, f), 'r', encoding='utf-8') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print("REPORTING CORPUS STATISTICS")
        print("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        print("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))

    def update_model(self, bow, label):
        """
        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        self.class_total_doc_counts[label] += 1
        for i in bow.keys():
            if i not in self.vocab:
                self.vocab.add(i)
            self.class_word_counts[label][i] += bow[i]
        self.class_total_word_counts[label] += len(bow)

    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """
        bow = tokenize_doc(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """
        Returns the most frequent n tokens for documents with class 'label'.
        I sort the class_word_counts dictionary base on the values of its keys.
        f is a dictionary containing top n repeated words in the class "lable".
        """
        self.class_word_counts[label] = dict(sorted(self.class_word_counts[label].items(), key=lambda item: item[1],
                                                    reverse=True))
        f, j = {}, 0
        for i in self.class_word_counts[label]:
            f[i] = self.class_word_counts[label][i]
            j += 1
            if j == n:
                break
        return f

    def p_word_given_label(self, word, label):
        """
        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        If the word is not in class_word_counts[label], an error occurs. To prevent these cases, the if statement check
        if this word is in our class_word_counts[label] or not.
        But the more accurate way is that we assume that we saw this word before. p_word_given_label_and_psuedocount
        operates base on this assumption.
        """
        if word in self.class_word_counts[label]:
            return self.class_word_counts[label][word]/self.class_total_word_counts[label]
        else:
            return 0

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        When there is a word which do not exist in one class or both classes,and error occurs in p_word_given_label
        function cause the word is not in our dictionary.
        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        x = 0
        if word in self.class_word_counts[label]:
            x = self.class_word_counts[label][word]
        d = len(self.vocab)
        return (alpha + x)/(self.class_total_word_counts[label] + (alpha*d))

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """
        ll = 0
        for i in bow.keys():
            ll += np.log(self.p_word_given_label_and_psuedocount(i, label, alpha))
        ll = round(ll, 3)
        return ll

    def log_prior(self, label):
        """
        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        return np.log(self.class_total_doc_counts[label] / (self.class_total_doc_counts[POS_LABEL] + self.class_total_doc_counts[NEG_LABEL]))

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        post = self.log_likelihood(bow, label, alpha) + self.log_prior(label)
        return post

    def classify(self, bow, alpha):
        """
        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)

        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """
        post_pos = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        post_neg = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        if post_pos > post_neg:
            return POS_LABEL
        else:
            return NEG_LABEL
        return None

    def likelihood_ratio(self, word, alpha):
        """

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        pos = self.p_word_given_label_and_psuedocount(word, POS_LABEL, alpha)
        neg = self.p_word_given_label_and_psuedocount(word, NEG_LABEL, alpha)
        return pos/neg

    def evaluate_classifier_accuracy(self, num_docs, alpha):
        """
        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right). Each time if the program classifies a doc correctly, right value is added by one.
        Accuracy = right / number of all documents
        """
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        right = 0
        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p, f), 'r', encoding='utf-8') as doc:
                    content = doc.read()
                    bow = tokenize_doc(content)
                    l = self.classify(bow, alpha)
                    if l == label:
                        right += 1
        print('Accuracy = %s' % (right/ (2*num_docs)))
        return right / (2 * num_docs)


def produce_hw4_results():

    d1 = "this sample doc has   words that  repeat repeat"
    bow = tokenize_doc(d1)

    assert bow['this'] == 1
    assert bow['sample'] == 1
    assert bow['doc'] == 1
    assert bow['has'] == 1
    assert bow['words'] == 1
    assert bow['that'] == 1
    assert bow['repeat'] == 2
    print()

    print("TOP 10 WORDS FOR CLASS " + POS_LABEL + " :")
    f = nb.top_n(POS_LABEL, 10)
    for tok in f:
        print('', tok, f[tok])
    print()
    print("TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :")
    f = nb.top_n(NEG_LABEL, 10)
    for tok in f:
        print('', tok, f[tok])
    print()
    print('p_word_given_label')
    print('P(fantastic | Pos):', nb.p_word_given_label('fantastic', POS_LABEL))
    print('P(fantastic | Neg):', nb.p_word_given_label('fantastic', NEG_LABEL))
    print('P(boring | Pos):', nb.p_word_given_label('boring', POS_LABEL))
    print('P(boring | Neg):', nb.p_word_given_label('boring', NEG_LABEL))
    print('p_word_given_label_and_psuedocount')
    print('P(fantastic | Pos):', nb.p_word_given_label_and_psuedocount('fantastic', POS_LABEL, 1.0))
    print('P(fantastic | Neg):', nb.p_word_given_label_and_psuedocount('fantastic', NEG_LABEL, 1.0))
    print('P(boring | Pos):', nb.p_word_given_label_and_psuedocount('boring', POS_LABEL, 1.0))
    print('P(boring | Neg):', nb.p_word_given_label_and_psuedocount('boring', NEG_LABEL, 1.0))
    print()
    print('LR(fantastic)', nb.likelihood_ratio('fantastic', 1.0))
    print('LR(boring)', nb.likelihood_ratio('boring', 1.0))
    print('LR(the)', nb.likelihood_ratio('the', 1.0))
    print('LR(to)', nb.likelihood_ratio('to', 1.0))
    print()


def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries.
    """

    import matplotlib.pyplot as plt

    plt.plot(psuedocounts, accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()


if __name__ == '__main__':
    nb = NaiveBayes()
    # nb.train_model()
    nb.train_model(num_docs=12500)
    produce_hw4_results()
    print('=============================================')
    print("Start Testing the Naive Bayes model on our test set:")
    alpha = [70, 50, 40, 30, 20, 10]
    acc = []
    for i in alpha:
        print('alpha = %s' % i)
        acc.append(nb.evaluate_classifier_accuracy(12500, i))
    plot_psuedocount_vs_accuracy(alpha, acc)

