from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf
import modeling
from tensorflow.python import pywrap_tensorflow
import numpy as np

chkp.print_tensors_in_checkpoint_file("output_dir_position_pretrain/model.ckpt-180000",tensor_name=None, all_tensors=False)

chkp.print_tensors_in_checkpoint_file("output_dir_dupe5_s4_3class/model_795_seq_direction.ckpt",
                                      tensor_name='cls/seq_direction/output_weights', all_tensors=False)
chkp.print_tensors_in_checkpoint_file("output_dir_dupe5_s4/model.ckpt-795000",
                                      tensor_name='cls/seq_direction/output_bias2', all_tensors=False)
with tf.variable_scope("cls/seq_direction"):
    output_weights2 = tf.get_variable(
        "output_weights2",
        shape=[1, 768],
        initializer=modeling.create_initializer(.02))
    output_bias2 = tf.get_variable(
        "output_bias2", shape=[1], initializer=tf.zeros_initializer())

# output_weights = chkp.print_tensors_in_checkpoint_file("uncased_L-12_H-768_A-12/bert_model.ckpt",
#                                                        tensor_name='cls/seq_relationship/output_weights',
#                                                        all_tensors=False)

# output_dir_position_pretrain_tf/model.ckpt-34000 bert/embeddings/dependency_embedding
# output_bias = chkp.print_tensors_in_checkpoint_file("uncased_L-12_H-768_A-12/bert_model.ckpt",
#                                                     tensor_name='cls/seq_relationship/output_bias', all_tensors=False)
#reader = pywrap_tensorflow.NewCheckpointReader("uncased_L-12_H-768_A-12/bert_model.ckpt")
#add new embeddinf
# reader.get_tensor('bert/embeddings/token_type_embeddings')
    ##add task 3
reader = pywrap_tensorflow.NewCheckpointReader("output_dir_position_pretrain/model.ckpt-180000")

output_bias = reader.get_tensor('bert/embeddings/dependency_embedding')
output_bias2 = output_bias[0]

output_weights = reader.get_tensor('cls/seq_direction/output_weights')
hidden_size = output_weights.shape[1]
output_weights2 = output_weights[0]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

new_saver = tf.train.import_meta_graph('output_dir_dupe5_s4/model.ckpt-795000.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('output_dir_dupe5_s4/'))

saver = tf.train.Saver()
saver.save(sess, 'output_dir_dupe5_s4_3class/model_795_seq_direction.ckpt')

#################change 3class
sess = tf.Session()
sess.run(tf.global_variables_initializer())

new_saver = tf.train.import_meta_graph('output_dir_dupe5_s4/model.ckpt-795000.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('output_dir_dupe5_s4/'))

list = tf.global_variables(scope="cls/seq_direction/")
for v in list:
    print(v)
tf.global_variables(scope="cls/seq_direction/")

output_weights = [v for v in tf.global_variables() if v.name == "cls/seq_direction/output_weights:0"][0]
# output_weights = sess.run(tf.concat([output_weights, [output_weights[0]]], 0))
assign_op = tf.assign(output_weights, tf.concat([output_weights, [output_weights[0]]], 0))


saver = tf.train.Saver()
saver.save(sess, 'output_dir_dupe5_s4_3class/model_795_seq_direction.ckpt')

output_bias = [v for v in tf.global_variables() if v.name == "cls/seq_direction/output_bias:0"][0]
output_bias_adam_m = [v for v in tf.global_variables() if v.name == "cls/seq_direction/output_bias/adam_m:0"][0]
tmp = output_bias_adam_m[0].eval(session=sess)
output_bias_adam_m = tf.concat([output_bias_adam_m, [tmp]], 0)
