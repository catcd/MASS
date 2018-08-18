import pickle

from constant import constants_bc5 as constant
from data_utils import *
from dataset.dataset_bc5 import BioCDataset

from dataset.dataset import merge_dataset


def main():
    # load vocabs
    vocab_words = load_vocab(constant.ALL_WORDS)
    vocab_chars = load_vocab(constant.ALL_CHARS)
    vocab_poses = load_vocab(constant.ALL_POSES)
    vocab_depends = load_vocab(constant.ALL_DEPENDS)
    vocab_wordnet_supersets, _ = load_wordnet_superset()

    print('build bc5 data')
    ds = merge_dataset(
        BioCDataset(
            'data/raw_data/bc5/bc5_data_original_lg.train.txt',
            vocab_words, vocab_chars, vocab_poses,
            vocab_depends, vocab_wordnet_supersets, constant
        ),
        BioCDataset(
            'data/raw_data/bc5/bc5_data_original_lg.dev.txt',
            vocab_words, vocab_chars, vocab_poses,
            vocab_depends, vocab_wordnet_supersets, constant
        ),
    )
    pickle.dump(ds, open(constant.PICKLE_DATA + 'bc5.fw.orig.train.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    ds = BioCDataset(
        'data/raw_data/bc5/bc5_data_original_lg.test.txt',
        vocab_words, vocab_chars, vocab_poses,
        vocab_depends, vocab_wordnet_supersets, constant
    )
    pickle.dump(ds, open(constant.PICKLE_DATA + 'bc5.fw.orig.test.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    print('build bc5 data done')


if __name__ == '__main__':
    main()
