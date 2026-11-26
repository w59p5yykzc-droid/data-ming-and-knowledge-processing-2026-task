import json
import os
import time
import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from lab_common import read_reviews, exclusion_keys, write_table

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent.parent
train_data = list(read_reviews(ROOT/"data"/"train_sample.csv"))
corpus = [r.tokens() for r in train_data]
print("样本数:",len(train_data))
exclude_files = [ROOT/"data"/"exclusions"/"dev.csv",ROOT/"data"/"exclusions"/"test.csv"]
ex_keys = exclusion_keys(exclude_files)
train_keys = set([r.key() for r in train_data])
overlap = train_keys & ex_keys

if len(overlap) > 0:
    print("数据重合:", len(overlap))
else:
    print("通过")

all_words = [w for sent in corpus for w in sent]
print("总词数:",len(all_words))
counts = Counter(all_words)
words_list = [w for w,c in counts.items() if c >= 2]
words_list.sort(key=lambda x: counts[x],reverse=True)
plot_words = words_list[:200]
windows = [2, 5]
word_pairs = [("good","great"),("good","bad"),("phone","battery"),("expensive","cheap"),("screen","battery")]
query_words = ["good","bad","great", "phone","battery","screen","price","quality","easy","poor"]
output_dir = ROOT/"outputs"
output_dir.mkdir(exist_ok=True)
res_comp = []
res_pairs = []
res_neighbors = []

for win in windows:
    print("\n开始训练window =",win)
    t0 = time.time()
    model = Word2Vec(corpus,vector_size=128,window=win,min_count=2,sg=0,seed=SEED,epochs=10,workers=1)
    cost_time = time.time() - t0
    print("词表大小:",len(model.wv), ";耗时:",round(cost_time,2),"秒")
    res_comp.append({"window":win,"vocab_size":len(model.wv),"cost_time":round(cost_time,2)})

    for w1,w2 in word_pairs:
        if w1 in model.wv and w2 in model.wv:
            sim = round(float(model.wv.similarity(w1,w2)),4)
            res_pairs.append({"window":win,"word1":w1,"word2":w2,"similarity":sim})
        else:
            res_pairs.append({"window":win,"word1":w1,"word2":w2,"similarity":"OOV"})

    for word in query_words:
        if word in model.wv:
            res = model.wv.most_similar(word,topn=5)

            for i,(nb,s) in enumerate(res,1):
                res_neighbors.append({"window":win,"query":word,"rank":i,"neighbor":nb,"similarity":round(float(s),4)})

    actual_plot_words = [w for w in plot_words if w in model.wv]
    vectors = model.wv[actual_plot_words]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    plt.figure(figsize=(10,8))
    plt.scatter(coords[:,0],coords[:,1],s=20,alpha=0.6)

    for i in range(min(30, len(actual_plot_words))):
        plt.text(coords[i,0], coords[i,1],actual_plot_words[i],fontsize=9)

    plt.title("PCA Window=" + str(win))
    plt.savefig(str(output_dir/("pca_win" + str(win) + ".png")))
    plt.close()

    if win == 5:
        model.save(str(output_dir/"w2v_window5.model"))
        np.save(str(output_dir / "embedding.npy"), model.wv.vectors)
        vocab = {
            word:index
            for index,word in enumerate(model.wv.index_to_key)
        }

        with open(output_dir/"vocab.json","w",encoding="utf-8") as f:
            json.dump(vocab,f,ensure_ascii=False,indent=2)

write_table(str(output_dir/"comparison.csv"),["window","vocab_size","cost_time"],res_comp)
write_table(str(output_dir/"word_pairs.csv"),["window","word1","word2","similarity"],res_pairs)
write_table(str(output_dir/"neighbors.csv"),["window","query","rank","neighbor","similarity"],res_neighbors)