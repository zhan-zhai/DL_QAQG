# coding=utf-8
# 数据处理工具
import numpy as np
import re
import jieba
from sklearn.preprocessing import OneHotEncoder


# 加载word2vec词表
def load_CN_word_vectors(path="./sgns.wiki.bigram-char"):
    word2vec = {}
    cnt = 1
    
    with open(path, encoding='utf-8') as f:
        words = f.readline().split(' ')
        total, dim = int(words[0]), int(words[1][: -1])
        
        while True:
            # print('\rreading line...%d/%d' % (cnt, total), end='')

            line = f.readline()
            words = line.split(' ')
            word2vec[words[0]] = np.array(words[1: -1])
            
            if not line:
                break
            cnt += 1
        f.close()
    
    return word2vec, dim


# 特殊符号处理函数
def punct_handling(inputstr, rep):
    remove_str = "[\s+\.:\!-\/_,$%\[\]^\)*(+\"\']+|[+——！1234567890·，。？?、~@#￥%……&*（），` | \" ~〜 “ １　２　３　４　５　６　７　８　９　０  ”【】 ※]+"
    punc = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.～∽～＞＝─=≡Σつ•̀Ω•́つ"
    # punc = punc.encode('utf-8').decode("utf-8")
    string1 = re.sub(r'[{}]'.format(remove_str), rep, inputstr)
    string1 = re.sub(r'[{}]'.format(punc), rep, string1)
    return string1


# jieba分词
def fenci(string):
    string = punct_handling(string, "")
    segs = jieba.cut(string, cut_all=False)
    # ret_string = ""
    # for seg in segs:
    #     ret_string += (seg+" ")
    # ret_string = ret_string.strip()
    return list(segs)


# 对训练数据中的问题文本和三元组文本做embedding，转化为由词向量构成的数据存储为.npy文件
def train_data_generate(path='./nlp-kbqa-training-data.txt'):
    #  处理问题文本
    question_vecs = []
    triple_vecs = []
    not_in_dict_word_q = []
    not_in_dict_word_t = []
    with open(path, encoding='utf-8') as f:
        word2vec, dim = load_CN_word_vectors()
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('<q'):
                question_str = line[line.find('>')+1:].strip()
                question_L = fenci(question_str)
                question_vec = []
                if len(question_L) == 0:
                    print(line)
                    f.readline()
                    continue
                for c in question_L:
                    if c in word2vec.keys():
                        question_vec.append(np.array(word2vec[c]).astype(np.float))
                    else:
                        # print(question_L, 'word:'+c+' not in word_dict')
                        not_in_dict_word_q.append(c)
                        question_vec.append(np.zeros(300))
                question_vecs.append(question_vec)

            elif line.startswith('<t'):
                triple_vec = []
                ent_str, att_str, val_str = line[line.find('>') + 1:].strip().split('|||')
                ent_str = ent_str.strip()
                att_str = att_str.strip()
                val_L = fenci(val_str.strip())

                if len(ent_str) == 0:
                    print(line)
                    question_vecs.pop()
                    continue

                if len(att_str) == 0:
                    print(line)
                    question_vecs.pop()
                    continue

                if len(val_L) == 0:
                    print(line)
                    question_vecs.pop()
                    continue

                if ent_str in word2vec.keys():
                    triple_vec.append(np.array(word2vec[ent_str]).astype(np.float))
                else:
                    triple_vec.append(np.zeros(300))
                if att_str in word2vec.keys():
                    triple_vec.append(np.array(word2vec[att_str]).astype(np.float))
                else:
                    triple_vec.append(np.zeros(300))

                for v in val_L:
                    if v in word2vec.keys():
                        triple_vec.append(np.array(word2vec[v]).astype(np.float))
                    else:
                        # print(question_L, 'word:'+c+' not in word_dict')
                        not_in_dict_word_t.append(v)
                        triple_vec.append(np.zeros(300))
                triple_vecs.append(triple_vec)

            else:
                continue

        print('question_len: ', len(question_vecs))
        print('triple_len: ', len(triple_vecs))
        print('len of unrecognized word in q: ', len(not_in_dict_word_q))
        print('len of unrecognized word in t: ', len(not_in_dict_word_t))

        for i in question_vecs:
            if len(i) == 0:
                print('vec len 0: ', i)
        for i in triple_vecs:
            if len(i) <= 2:
                print('len triple', len(i))
        if len(question_vecs) != len(triple_vecs):
            print('len not equal', len(question_vecs), ' ', len(triple_vecs))
        np.save('train_question_vecs', question_vecs)
        np.save('train_triple_vecs', triple_vecs)


# 对三元组中的实体和属性做独热编码
def ent_att_OneHot():
    train_path = 'nlp-kbqa-training-data.txt'
    test_path = 'nlp-kbqa-testing-data.txt'
    ent_L_train = []
    att_L_train = []
    ent_L_all = []
    att_L_all = []
    with open(train_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('<t'):
                ent_str, att_str, val_str = line[line.find('>') + 1:].strip().split('|||')
                ent_str = ent_str.strip()
                att_str = att_str.strip()
                ent_L_train.append(ent_str)
                att_L_train.append(att_str)
                ent_L_all.append(ent_str)
                att_L_all.append(att_str)
            else:
                continue
    print(len(ent_L_all))
    print(len(att_L_all))

    with open(test_path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('<t'):
                ent_str, att_str, val_str = line[line.find('>') + 1:].strip().split('|||')
                ent_str = ent_str.strip()
                att_str = att_str.strip()
                ent_L_all.append(ent_str)
                att_L_all.append(att_str)
            else:
                continue
    print(len(ent_L_all))
    print(len(att_L_all))

    ent_L_all = np.array(ent_L_all).reshape(-1, 1)
    ent_L_train = np.array(ent_L_train).reshape(-1, 1)
    att_L_all = np.array(att_L_all).reshape(-1, 1)
    att_L_train = np.array(att_L_train).reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    ent_OneHotEncoded = onehot_encoder.fit(ent_L_all)  # 18746种
    ent_L_train = ent_OneHotEncoded.transform(ent_L_train)
    print(len(ent_L_train))
    print(len(ent_L_train[0]))
    # print(ent_L_train[0])
    att_OneHotEncoded = onehot_encoder.fit(att_L_all)  #
    att_L_train = att_OneHotEncoded.transform(att_L_train)
    print(len(att_L_train))
    print(len(att_L_train[0]))
    # print(att_L_train[0])

    ent_att_onehot = np.hstack([ent_L_train, att_L_train])
    print(ent_att_onehot.shape)
    np.save('ent_att_onehot', ent_att_onehot)


# 根据给定的词向量，将其转化为对应的词
def vec2word(embedding):
    if len(embedding) != 300:
        print('input len x')
    word2vec, dim = load_CN_word_vectors()
    words = list(word2vec.keys())
    vecs_str = list(word2vec.values())
    print(len(vecs_str))
    words_new = []
    vecs_float = []
    for i, v in enumerate(vecs_str):
        v_float = [float(x) for x in v]
        if len(v_float) != 300:
            # print('vec len is not 300 ', len(v_float))
            # print(words[i])
            continue
        words_new.append(words[i])
        vecs_float.append(v_float)
    vecs_float = np.array(vecs_float)
    min_index = 0
    min_dis = 0.5 * np.sum((vecs_float[0] - embedding) ** 2)
    for i, v in enumerate(vecs_float):
        d = 0.5 * np.sum((v - embedding) ** 2)
        if d < min_dis:
            min_index = i
            min_dis = d
    return words_new[min_index]


# 根据给定的实体和属性，在知识库中寻找对应属性值
def search_answer(entity, attribute):
    print('finding ', entity, ' ', attribute)
    path = './nlp-kbqa-training-data.txt'
    KB = {}
    # 建立知识库的索引
    with open(path, encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('<t'):
                ent_str, att_str, val_str = line[line.find('>') + 1:].strip().split('|||')
                ent_str = ent_str.strip()
                att_str = att_str.strip()
                val_str = val_str.strip()

                if len(ent_str) == 0:
                    # print(line)
                    continue
                if len(att_str) == 0:
                    # print(line)
                    continue
                if len(val_str) == 0:
                    # print(line)
                    continue

                if ent_str not in KB.keys():
                    KB[ent_str] = {att_str: val_str}
                else:
                    if att_str not in KB[ent_str].keys():
                        KB[ent_str][att_str] = val_str
                    # print('dict error', ent_str, ' ', att_str, ' ', val_str)
                    # print(KB[ent_str][att_str])
            else:
                continue
    if entity not in KB.keys() or attribute not in KB[entity].keys():
        print('can not find the answer in KB')
        return ''
    return KB[entity][attribute]


if __name__ == "__main__":
    train_data_generate()
    # ent_att_OneHot()
    # print(search_answer('机械设计基础', '作者'))
