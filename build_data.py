import os

from trial.semeval_build_data import main as run_semeval
from trial.bc5_build_data import main as run_bc5
from utils import Timer
from data_utils import export_trimmed_fasttext_vectors
from data_utils import load_vocab
from constant import constants_semeval as sem_constants
from constant import constants_bc5 as bc5_constants

if __name__ == '__main__':
    timer = Timer()
    timer.start("Building data...")
    print('Build trimmed embeddings')
    bc5_vocab = load_vocab(bc5_constants.ALL_WORDS)
    sem_vocab = load_vocab(sem_constants.ALL_WORDS)

    export_trimmed_fasttext_vectors(bc5_vocab, 'data/w2v_model/bc5_fasttext.npz')
    export_trimmed_fasttext_vectors(sem_vocab, 'data/w2v_model/semeval_fasttext.npz')

    print('Build pickle file')
    os.makedirs('data/pickle', exist_ok=True)

    run_semeval()
    run_bc5()

    timer.stop()
