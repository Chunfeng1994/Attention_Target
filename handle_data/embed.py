# 这个是extendEmbedding
from handle_data.CreatVocab import *
def create_vocab_embs(embfile,src_dic):
    print(embfile)
    embedding_dim = -1
    embed_word_count = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            if embed_word_count < 1:
                values = line.split()
                embedding_dim = len(values) - 1
            embed_word_count += 1
    print('\nTotal words: ' + str(embed_word_count))
    print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

    find_count = 0
    embeddings = np.zeros((len(src_dic.i2w), embedding_dim))
    # ii = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            values = line.split(' ')
            if values[0] in src_dic.w2i:
                vector = np.array(values[1:], dtype='float64')
                embeddings[src_dic.w2i[values[0]]] = vector

                embeddings[UNK] += vector

                find_count += 1
            # ii+= 1
            # print(ii)

    print("The number of vocab word find in extend embedding is: ", str(find_count))
    print("The number of all vocab is: ", str(len(src_dic.w2i)))
    embeddings[UNK] = embeddings[UNK] / find_count
    embeddings = embeddings / np.std(embeddings)
    print(embeddings)
    not_find = len(src_dic.w2i) - find_count
    oov_ratio = float(not_find / len(src_dic.w2i))

    print('oov ratio: {:.4f}'.format(oov_ratio))

    return embeddings