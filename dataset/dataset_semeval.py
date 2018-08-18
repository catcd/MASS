from dataset.dataset import Dataset


class SemEvalDataset(Dataset):
    def parse_raw(self, raw_data):
        sdp = []
        rela = []
        all_words = []
        all_relations = []
        all_directions = []
        all_poses = []
        all_labels = []
        all_lens = []
        all_ids = []

        for line in raw_data:
            l = line.strip().split()
            if len(l) > 2:
                sdp.append(l)
            elif len(l) == 2:
                rela = l
            else:
                label = rela[1]
                if label and sdp:
                    sd = sdp[0]
                    words = []
                    poses = []
                    relations = []
                    directions = []
                    for node in sd[1:]:
                        node = node.rsplit('/', 1)
                        if len(node) == 2:
                            word = self.constant.UNK if node[0] == '' else node[0]
                            pos = 'NN' if node[-1] == '' else node[-1]

                            words.append(word)
                            poses.append(pos)
                        else:
                            dependency = node[0]
                            r = '(' + dependency[3:]
                            d = dependency[1]
                            r = r.split(':', 1)[0] + ')' if ':' in r else r
                            relations.append(r)
                            directions.append(d)

                    all_words.append(words)
                    all_lens.append(len(words))
                    all_relations.append(relations)
                    all_directions.append(directions)
                    all_poses.append(poses)
                    all_labels.append([label])
                    all_ids.append(rela[0])

                    sdp = []
                else:
                    print(rela[0])
        return all_words, all_labels, all_lens, all_poses, all_relations, all_directions, all_ids
