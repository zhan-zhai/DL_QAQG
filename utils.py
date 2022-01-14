import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def vector_padding(vecs):
    l_seq = []
    cnt, total = 0, vecs.shape[0]

    for vi in vecs:
        print("\rpadding vectors  %d/%d..." % (cnt + 1, total), end='')        

        vi = torch.tensor(vi, dtype=torch.float32)
        if vi.size()[0] != 0:
            l_seq.append(vi)
        else:
            print("\n",vi.shape, "idx =", cnt)
        cnt += 1

    pad_vecs = pad_sequence(l_seq, batch_first=True)
    print("Done!")
    return pad_vecs


def extract_entity_relation_pair(triple_vecs):
        en_vecs, re_vecs = [[]], [[]]
        cnt, total = 0, triple_vecs.shape[0]

        for ti in triple_vecs:
            print("\rextracting pairs %d/%d..." % (cnt + 1, total), end="")

            eni = ti[0]
            rei = ti[1]

            if eni.shape[0] == 0 or rei.shape[0] == 0:
                print("\n", eni.shape, rei.shape, "idx =", cnt)
                continue

            en_vecs[0].append(eni)
            re_vecs[0].append(rei)

            cnt += 1
        en_vecs = np.array(en_vecs)
        re_vecs = np.array(re_vecs)

        pair_vecs = np.concatenate((en_vecs, re_vecs), axis=0)
        pair_vecs = pair_vecs.reshape(pair_vecs.shape[1], pair_vecs.shape[0], pair_vecs.shape[2])
        print("Done!")
        return pair_vecs

    