import pickle

from constant import constants_semeval as constant
from data_utils import *
from dataset.dataset_semeval import SemEvalDataset


def main():
    # load vocabs
    vocab_words = load_vocab(constant.ALL_WORDS)
    vocab_chars = load_vocab(constant.ALL_CHARS)
    vocab_poses = load_vocab(constant.ALL_POSES)
    vocab_depends = load_vocab(constant.ALL_DEPENDS)
    vocab_wordnet_supersets, _ = load_wordnet_superset()

    print('build semeval pickle data')
    ds = SemEvalDataset(
        'data/raw_data/semeval/semeval_data_original_lg.test.txt',
        vocab_words, vocab_chars, vocab_poses,
        vocab_depends, vocab_wordnet_supersets, constant
    )
    pickle.dump(ds, open(constant.PICKLE_DATA + 'sem.fw.orig.test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    ds = SemEvalDataset(
        'data/raw_data/semeval/semeval_data_original_lg.train.txt',
        vocab_words, vocab_chars, vocab_poses,
        vocab_depends, vocab_wordnet_supersets, constant
    )
    pickle.dump(ds, open(constant.PICKLE_DATA + 'sem.fw.orig.train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    print('build semeval data done')


if __name__ == '__main__':
    main()
