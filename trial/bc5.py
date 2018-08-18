import pickle

import json

import copy

from constant import constants_bc5 as constant
from data_utils import *
from model.lstm_cnn_model import LstmCnn
from evaluate.bc5 import evaluate_bc5, evaluate_bc5_intra


def main():
    # train on full
    train = pickle.load(open(constant.PICKLE_DATA + 'bc5.fw.orig.train.pickle', 'rb'))

    test = pickle.load(open(constant.PICKLE_DATA + 'bc5.fw.orig.test.pickle', 'rb'))

    validation = pickle.load(open(constant.PICKLE_DATA + 'bc5.fw.orig.train.pickle', 'rb')).one_over_ten()

    # get pre trained embeddings
    fasttext_embeddings = get_trimmed_w2v_vectors(constant.TRIMMED_FASTTEXT_W2V)
    _, wordnet_superset_embeddings = load_wordnet_superset()

    model = LstmCnn(
        model_name=constant.MODEL_NAMES.format('bc5', constant.JOB_IDENTITY),
        embeddings_fasttext=fasttext_embeddings,
        embeddings_wordnet_superset=wordnet_superset_embeddings,
        batch_size=128,
        constants=constant,
    )

    # train, evaluate and interact
    model.build()
    model.load_data(train=train, validation=validation)
    model.run_train(epochs=constant.EPOCHS, early_stopping=constant.EARLY_STOPPING, patience=constant.PATIENCE)

    # tunable
    CID_PRED_LABEL = {
        'CID',      # for fw
        'CID(r)',   # for bw
    }
    CID_PRED = {constant.ALL_LABELS.index(i) for i in CID_PRED_LABEL}

    answer = []
    identities = test.identities
    y_pred = model.predict(test)
    for i in range(len(y_pred)):
        if y_pred[i] in CID_PRED and (identities[i][0], identities[i][1], 'CID') not in answer:
            answer.append((identities[i][0], identities[i][1], 'CID'))
    print('result: abstract: ', evaluate_bc5(answer), 'intra', evaluate_bc5_intra(answer))

    chemical_rule = json.load(open('data/bc5_chemical_rule.txt', 'r'))

    answer2 = copy.deepcopy(answer)
    for pmid in chemical_rule:
        if not __is_in(pmid, answer2):
            new = chemical_rule[pmid]['title'] if len(chemical_rule[pmid]['title']) != 0 else chemical_rule[pmid]['frequency']
            for pair in new:
                answer2.append((pmid, pair, 'CID'))
    print('result after apply chemical rule 1/2: abstract: ', evaluate_bc5(answer2))


def __is_in(pmid, answer):
    for a in answer:
        if pmid == a[0]:
            return True
    return False


if __name__ == '__main__':
    main()
