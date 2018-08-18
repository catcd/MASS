import argparse

# new: taking direction into account
ALL_LABELS = ['NONE', 'CID', 'CID(r)']
ALL_LABELS_2 = ['NONE', 'CID']

LABEL_2_LABEL2_MAP = {
    'NONE': ALL_LABELS_2.index('NONE'),
    'CID': ALL_LABELS_2.index('CID'),
    'CID(r)': ALL_LABELS_2.index('CID'),
}

LABEL_2_LABELB_MAP = {
    'NONE': ALL_LABELS.index('NONE'),
    'CID': ALL_LABELS.index('CID(r)'),
    'CID(r)': ALL_LABELS.index('CID'),
}

UNK = '$UNK$'

parser = argparse.ArgumentParser(description='Multi-channel biLSTM-CNN for relation extraction')

parser.add_argument('-i', help='Job identity', type=int, default=0)

parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=10)

parser.add_argument('-rus', help='Random under sampling number', type=int, default=0)
parser.add_argument('-ros', help='Random over sampling number', type=int, default=0)
parser.add_argument('-rss', help='Random seed for re-sampler', type=int, default=0)

parser.add_argument('-msl', help='Trimmed max sentence length', type=int, default=0)

parser.add_argument('-cnn1', help='Number of CNN region size 1 filters', type=int, default=256)
parser.add_argument('-cnn2', help='Number of CNN region size 2 filters', type=int, default=128)
parser.add_argument('-cnn3', help='Number of CNN region size 3 filters', type=int, default=0)

parser.add_argument('-ft', help='Number of output fastText w2v embedding LSTM dimension', type=int, default=100)

parser.add_argument('-wns', help='Number of output Wordnet superset LSTM dimension', type=int, default=25)

parser.add_argument('-char', help='Number of output character embedding LSTM dimension', type=int, default=25)
parser.add_argument('-chari', help='Number of input character embedding LSTM dimension', type=int, default=85)

parser.add_argument('-pos', help='Number of output POS tag LSTM dimension', type=int, default=50)
parser.add_argument('-posi', help='Number of input POS tag LSTM dimension', type=int, default=57)

parser.add_argument('-rel', help='Number of output dependency relation LSTM dimension', type=int, default=50)
parser.add_argument('-reli', help='Number of input dependency relation LSTM dimension', type=int, default=100)
parser.add_argument('-dir', help='Number of dependency direction embedding dimension', type=int, default=50)

parser.add_argument('-hd', help='Hidden layer configurations default \'128,128\'', type=str, default='128,128')

parser.add_argument('-a', help='Alpha ratio default 0.55', type=float, default=0.55)

opt = parser.parse_args()
print('Running bc5 opt: {}'.format(opt))

JOB_IDENTITY = opt.i

EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
PATIENCE = opt.p

RANDOM_SEED_RE_SAMPLE = opt.rss

USE_ROS = False if opt.ros == 0 else True
RANDOM_OVER_SAMPLER = opt.ros

USE_RUS = False if opt.rus == 0 else True
RANDOM_UNDER_SAMPLER = opt.rus

MAX_SENT_LENGTH = float('inf') if opt.msl < 2 else opt.msl

USE_CNN_REGION_SIZE_1 = False if opt.cnn1 == 0 else True
CNN_REGION_SIZE_1_FILTER = opt.cnn1
USE_CNN_REGION_SIZE_2 = False if opt.cnn2 == 0 else True
CNN_REGION_SIZE_2_FILTER = opt.cnn2
USE_CNN_REGION_SIZE_3 = False if opt.cnn3 == 0 else True
CNN_REGION_SIZE_3_FILTER = opt.cnn3

USE_FASTTEXT = False if opt.ft == 0 else True
INPUT_FASTTEXT_DIM = 300
OUTPUT_LSTM_FASTTEXT_DIM = opt.ft

USE_WORDNET_SUPERSET = False if opt.wns == 0 else True
INPUT_WORDNET_SUPERSET_DIM = 45
OUTPUT_LSTM_WORDNET_SUPERSET_DIM = opt.wns

USE_CHAR = False if opt.char == 0 else True
NCHARS = 87
INPUT_LSTM_CHAR_DIM = opt.chari
OUTPUT_LSTM_CHAR_DIM = opt.char

USE_POS = False if opt.pos == 0 else True
NPOS = 57
INPUT_LSTM_POS_DIM = opt.posi
OUTPUT_LSTM_POS_DIM = opt.pos

USE_RELATION = False if opt.rel == 0 else True
NRELATIONS = 190
INPUT_LSTM_RELATION_DIM = opt.reli
OUTPUT_LSTM_RELATION_DIM = opt.rel

USE_DIRECTION = False if opt.dir == 0 else True
NDIRECTIONS = 3
DIRECTION_EMBEDDING_DIM = opt.dir

HIDDEN_LAYERS = list(map(int, opt.hd.split(','))) if opt.hd else []

ALPHA = opt.a

DATA = 'data/'
RAW_DATA = DATA + 'raw_data/'
PARSED_DATA = DATA + 'parsed_data/'
PICKLE_DATA = 'data/pickle/'

ALL_WORDS = PARSED_DATA + 'bc5_all_words.txt'
ALL_CHARS = PARSED_DATA + 'bc5_all_chars.txt'
ALL_POSES = PARSED_DATA + 'all_pos.txt'
ALL_DEPENDS = PARSED_DATA + 'all_depend.txt'

W2V_DATA = DATA + 'w2v_model/'
TRIMMED_FASTTEXT_W2V = W2V_DATA + 'bc5_fasttext.npz'

TRAINED_MODELS = DATA + 'trained_models/bc5/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
