from dataset.dataset import Dataset


class BioCDataset(Dataset):
    def parse_raw(self, raw_data):
        all_words = []
        all_relations = []
        all_directions = []
        all_poses = []
        all_labels = []
        all_lens = []
        all_identities = []
        pmid = ''
        for line in raw_data:
            l = line.strip().split()
            if len(l) == 1:
                pmid = l[0]
            else:
                try:
                    pair = l[0]
                    label = l[1]
                    if label:
                        joint_sdp = ' '.join(l[2:])
                        sdps = joint_sdp.split("-PUNC-")
                        for sdp in sdps:
                            nodes = sdp.split()
                            words = []
                            poses = []
                            relations = []
                            directions = []

                            for node in nodes:
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
                            all_identities.append((pmid, pair))
                    else:
                        print(l)
                except Exception as e:
                    print(e.__str__())

        return all_words, all_labels, all_lens, all_poses, all_relations, all_directions, all_identities
