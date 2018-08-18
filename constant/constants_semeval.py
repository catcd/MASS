import argparse

ALL_LABELS = [
    'Cause-Effect(e1,e2)', 'Component-Whole(e1,e2)', 'Content-Container(e1,e2)', 'Entity-Destination(e1,e2)',
    'Entity-Origin(e1,e2)', 'Instrument-Agency(e1,e2)', 'Member-Collection(e1,e2)', 'Message-Topic(e1,e2)', 'Product-Producer(e1,e2)',
    'Other',
    'Cause-Effect(e2,e1)', 'Component-Whole(e2,e1)', 'Content-Container(e2,e1)', 'Entity-Destination(e2,e1)',
    'Entity-Origin(e2,e1)', 'Instrument-Agency(e2,e1)', 'Member-Collection(e2,e1)', 'Message-Topic(e2,e1)', 'Product-Producer(e2,e1)'
]
ALL_LABELS_2 = [
    'Cause-Effect', 'Component-Whole', 'Content-Container', 'Entity-Destination', 'Entity-Origin', 'Instrument-Agency',
    'Member-Collection', 'Message-Topic', 'Product-Producer', 'Other',
]

LABEL_2_LABEL2_MAP = {
    'Cause-Effect(e1,e2)': ALL_LABELS_2.index('Cause-Effect'),
    'Component-Whole(e1,e2)': ALL_LABELS_2.index('Component-Whole'),
    'Content-Container(e1,e2)': ALL_LABELS_2.index('Content-Container'),
    'Entity-Destination(e1,e2)': ALL_LABELS_2.index('Entity-Destination'),
    'Entity-Origin(e1,e2)': ALL_LABELS_2.index('Entity-Origin'),
    'Instrument-Agency(e1,e2)': ALL_LABELS_2.index('Instrument-Agency'),
    'Member-Collection(e1,e2)': ALL_LABELS_2.index('Member-Collection'),
    'Message-Topic(e1,e2)': ALL_LABELS_2.index('Message-Topic'),
    'Product-Producer(e1,e2)': ALL_LABELS_2.index('Product-Producer'),
    'Other': ALL_LABELS_2.index('Other'),
    'Cause-Effect(e2,e1)': ALL_LABELS_2.index('Cause-Effect'),
    'Component-Whole(e2,e1)': ALL_LABELS_2.index('Component-Whole'),
    'Content-Container(e2,e1)': ALL_LABELS_2.index('Content-Container'),
    'Entity-Destination(e2,e1)': ALL_LABELS_2.index('Entity-Destination'),
    'Entity-Origin(e2,e1)': ALL_LABELS_2.index('Entity-Origin'),
    'Instrument-Agency(e2,e1)': ALL_LABELS_2.index('Instrument-Agency'),
    'Member-Collection(e2,e1)': ALL_LABELS_2.index('Member-Collection'),
    'Message-Topic(e2,e1)': ALL_LABELS_2.index('Message-Topic'),
    'Product-Producer(e2,e1)': ALL_LABELS_2.index('Product-Producer')
}

LABEL_2_LABELB_MAP = {
    'Cause-Effect(e1,e2)': ALL_LABELS.index('Cause-Effect(e2,e1)'),
    'Component-Whole(e1,e2)': ALL_LABELS.index('Component-Whole(e2,e1)'),
    'Content-Container(e1,e2)': ALL_LABELS.index('Content-Container(e2,e1)'),
    'Entity-Destination(e1,e2)': ALL_LABELS.index('Entity-Destination(e2,e1)'),
    'Entity-Origin(e1,e2)': ALL_LABELS.index('Entity-Origin(e2,e1)'),
    'Instrument-Agency(e1,e2)': ALL_LABELS.index('Instrument-Agency(e2,e1)'),
    'Member-Collection(e1,e2)': ALL_LABELS.index('Member-Collection(e2,e1)'),
    'Message-Topic(e1,e2)': ALL_LABELS.index('Message-Topic(e2,e1)'),
    'Product-Producer(e1,e2)': ALL_LABELS.index('Product-Producer(e2,e1)'),
    'Other': ALL_LABELS.index('Other'),
    'Cause-Effect(e2,e1)': ALL_LABELS.index('Cause-Effect(e1,e2)'),
    'Component-Whole(e2,e1)': ALL_LABELS.index('Component-Whole(e1,e2)'),
    'Content-Container(e2,e1)': ALL_LABELS.index('Content-Container(e1,e2)'),
    'Entity-Destination(e2,e1)': ALL_LABELS.index('Entity-Destination(e1,e2)'),
    'Entity-Origin(e2,e1)': ALL_LABELS.index('Entity-Origin(e1,e2)'),
    'Instrument-Agency(e2,e1)': ALL_LABELS.index('Instrument-Agency(e1,e2)'),
    'Member-Collection(e2,e1)': ALL_LABELS.index('Member-Collection(e1,e2)'),
    'Message-Topic(e2,e1)': ALL_LABELS.index('Message-Topic(e1,e2)'),
    'Product-Producer(e2,e1)': ALL_LABELS.index('Product-Producer(e1,e2)')
}

UNK = '$UNK$'

parser = argparse.ArgumentParser(description='Multi-channel biLSTM-CNN for relation extraction')

parser.add_argument('-i', help='Job identity', type=int, default=0)

parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=100)

parser.add_argument('-rus', help='Random under sampling number', type=int, default=0)
parser.add_argument('-ros', help='Random over sampling number', type=int, default=0)
parser.add_argument('-rss', help='Random seed for re-sampler', type=int, default=0)

parser.add_argument('-msl', help='Trimmed max sentence length', type=int, default=0)

parser.add_argument('-cnn1', help='Number of CNN region size 1 filters', type=int, default=128)
parser.add_argument('-cnn2', help='Number of CNN region size 2 filters', type=int, default=64)
parser.add_argument('-cnn3', help='Number of CNN region size 3 filters', type=int, default=32)

parser.add_argument('-ft', help='Number of output fastText w2v embedding LSTM dimension', type=int, default=100)

parser.add_argument('-wns', help='Number of output Wordnet superset LSTM dimension', type=int, default=50)

parser.add_argument('-char', help='Number of output character embedding LSTM dimension', type=int, default=50)
parser.add_argument('-chari', help='Number of input character embedding LSTM dimension', type=int, default=85)

parser.add_argument('-pos', help='Number of output POS tag LSTM dimension', type=int, default=25)
parser.add_argument('-posi', help='Number of input POS tag LSTM dimension', type=int, default=57)

parser.add_argument('-rel', help='Number of output dependency relation LSTM dimension', type=int, default=50)
parser.add_argument('-reli', help='Number of input dependency relation LSTM dimension', type=int, default=100)
parser.add_argument('-dir', help='Number of dependency direction embedding dimension', type=int, default=100)

parser.add_argument('-hd', help='Hidden layer configurations default \'128,128\'', type=str, default='128,128')

parser.add_argument('-a', help='Alpha ratio default 0.5', type=float, default=0.5)

opt = parser.parse_args()
print('Running semeval opt: {}'.format(opt))

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
NCHARS = 85
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

ALL_WORDS = PARSED_DATA + 'semeval_all_words.txt'
ALL_CHARS = PARSED_DATA + 'semeval_all_chars.txt'
ALL_POSES = PARSED_DATA + 'all_pos.txt'
ALL_DEPENDS = PARSED_DATA + 'all_depend.txt'

W2V_DATA = DATA + 'w2v_model/'
TRIMMED_FASTTEXT_W2V = W2V_DATA + 'semeval_fasttext.npz'

TRAINED_MODELS = DATA + 'trained_models/semeval/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
