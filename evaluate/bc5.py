from collections import Counter
from constant import constants_bc5 as constants


def __prf_bc5(tp, predict, golden):
    if tp == 0:
        return 0.0, 0.0, 0.0
    else:
        p = 1.0 * tp / predict
        r = 1.0 * tp / golden
        f1 = 2 * p * r / (p + r)
        return p, r, f1


def __evaluate_bc5(eval_data, eval_map):
    # count TP
    tp_map = {
        k: 0 for k in constants.ALL_LABELS_2
    }
    for pmid, pair, rel in set(eval_data):
        arg1, arg2 = pair.split('_')

        if (pmid, arg1, arg2) in eval_map:
            label2 = constants.ALL_LABELS_2[constants.LABEL_2_LABEL2_MAP[rel]]
            if eval_map[(pmid, arg1, arg2)] == label2:
                tp_map[label2] += 1
        elif (pmid, arg2, arg1) in eval_map:
            label2 = constants.ALL_LABELS_2[
                constants.LABEL_2_LABEL2_MAP[
                    constants.ALL_LABELS[
                        constants.LABEL_2_LABELB_MAP[rel]
                    ]
                ]
            ]
            if eval_map[(pmid, arg2, arg1)] == label2:
                tp_map[label2] += 1
    # count predict
    pred_map = Counter(constants.ALL_LABELS_2[constants.LABEL_2_LABEL2_MAP[rel]] for _, _, rel in set(eval_data))
    for k in constants.ALL_LABELS_2:
        if k not in pred_map:
            pred_map[k] = 0
    # count golden
    gold_map = Counter(eval_map[k] for k in eval_map)
    for k in constants.ALL_LABELS_2:
        if k not in gold_map:
            gold_map[k] = 0
    # calculate prf
    ret = [__prf_bc5(tp_map[label2], pred_map[label2], gold_map[label2]) for label2 in constants.ALL_LABELS_2]
    return ret[1]


def evaluate_bc5(eval_data):
    """
    :param list of (str, str, str) eval_data: pmid, pair, relation (2n+1)
    :return:
    """
    # load evaluate map
    eval_map = {}  # dict (pmid, e1, e2) => relation (n + 1)
    f = open('data/bc5_evaluate.txt')
    for line in f:
        pmid, rel, e1, e2 = line.strip().split()
        if (pmid, e1, e2) in eval_map and eval_map[(pmid, e1, e2)] != rel:
            print((pmid, e1, e2))
        eval_map[(pmid, e1, e2)] = rel

    return __evaluate_bc5(eval_data, eval_map)


def evaluate_bc5_intra(eval_data):
    """
    :param list of (str, str, str) eval_data: pmid, pair, relation (2n+1)
    :return:
    """
    # load evaluate map
    eval_map = {}  # dict (pmid, e1, e2) => relation (n + 1)
    f = open('data/bc5_evaluate_intra.txt')
    for line in f:
        pmid, rel, e1, e2 = line.strip().split()
        if rel.endswith('(r)'):
            eval_map[(pmid, e2, e1)] = rel[:-3]
        else:
            eval_map[(pmid, e1, e2)] = rel

    return __evaluate_bc5(eval_data, eval_map)
