import numpy as np
import fastText


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.

        FIX: Have you tried running python build_data first?
        This will build vocab file from your train, test and dev sets and
        trim your word vectors.""".format(filename)

        super(MyIOError, self).__init__(message)


def export_trimmed_fasttext_vectors(vocab, trimmed_filename, dim=300):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
        :param bin:
    """
    # embeddings contains embedding for the pad_tok as well
    embeddings = np.zeros([len(vocab) + 1, dim])

    m = fastText.load_model('data/w2v_model/wiki.en.bin')

    for word in vocab:
        if word == '$UNK$':
            continue

        embedding = m.get_word_vector(word)
        word_idx = vocab[word]
        embeddings[word_idx] = embedding

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx + 1  # preserve idx 0 for pad_tok

    except IOError:
        raise MyIOError(filename)
    return d


def load_wordnet_superset():
    """
    Returns:
        d: dict[word] = index
        embeddings: list of list embedding
    """
    d = dict()
    embeddings = [np.zeros(45, dtype=float)]
    with open('data/knowledge_base/wordnet_superset.txt') as f:
        for idx, line in enumerate(f):
            word, vec = line.strip().split('\t', 1)
            d[word] = idx + 1  # preserve idx 0 for pad_tok

            embedding = list(map(float, vec.split()))
            embeddings.append(np.array(embedding))

    return d, np.array(embeddings)
