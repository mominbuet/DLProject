from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

reader = pywrap_tensorflow.NewCheckpointReader("uncased_L-12_H-768_A-12/bert_model.ckpt")
word_embedding = reader.get_tensor('bert/embeddings/word_embeddings')
dependency_embedding = np.load('gc_save_768_1.npy')
# normalize
dependency_embedding /= np.max(np.abs(dependency_embedding), axis=0)
import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file="uncased_L-12_H-768_A-12/vocab.txt", do_lower_case=False)
map = dict()
with open("mapping_8000.txt", "r") as f:
    lines = f.readline()
    while lines:
        vals = lines.split(":")
        map[int(vals[0])] = int(vals[1])
        lines = f.readline()

bert_dependency_embedding = np.zeros(shape=(len(tokenizer.vocab),dependency_embedding.shape[1] ))
for i in range(0, len(tokenizer.vocab)):
    if i in map.keys():
        bert_dependency_embedding[i] = dependency_embedding[map[i]]
    else:
        bert_dependency_embedding[i] = np.random.normal(loc=np.mean(dependency_embedding),scale=np.std(dependency_embedding),size=dependency_embedding.shape[1])

np.save('bert_dependency_embedding_vae.npy', bert_dependency_embedding)
