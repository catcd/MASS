import pickle

from constant import constants_semeval as constant
from data_utils import *
from model.lstm_cnn_model import LstmCnn


def main():
    # train on train
    train = pickle.load(open(constant.PICKLE_DATA + 'sem.fw.orig.train.pickle', 'rb'))

    # test on test
    test = pickle.load(open(constant.PICKLE_DATA + 'sem.fw.orig.test.pickle', 'rb'))

    # validation on test
    validation = pickle.load(open(constant.PICKLE_DATA + 'sem.fw.orig.train.pickle', 'rb')).one_over_ten()

    # get pre trained embeddings
    fasttext_embeddings = get_trimmed_w2v_vectors(constant.TRIMMED_FASTTEXT_W2V)
    _, wordnet_superset_embeddings = load_wordnet_superset()

    model = LstmCnn(
        model_name=constant.MODEL_NAMES.format('semeval', constant.JOB_IDENTITY),
        embeddings_fasttext=fasttext_embeddings,
        embeddings_wordnet_superset=wordnet_superset_embeddings,
        batch_size=128,
        constants=constant,
    )

    # train, evaluate and interact
    model.build()
    model.load_data(train=train, validation=validation)
    model.run_train(epochs=constant.EPOCHS, early_stopping=constant.EARLY_STOPPING, patience=constant.PATIENCE)

    identities = test.identities
    y_pred = model.predict(test)
    of = open('data/output/answer-{}'.format(constant.JOB_IDENTITY), 'w')

    for i in range(len(y_pred)):
        of.write('{}\t{}\n'.format(identities[i], constant.ALL_LABELS[y_pred[i]]))

    of.close()


if __name__ == '__main__':
    main()
