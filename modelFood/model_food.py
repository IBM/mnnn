
import numpy as np

from .base_model import BaseModel
from .data_utils import minibatches, pad_sequences
from .general_utils import Progbar

np.set_printoptions(threshold=np.nan)

from modelFood import NlabelCell
import os
import math
import joblib
import argparse
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str,default="food")
parser.add_argument('--dataset', type=str,default="food")
parser.add_argument('--log_dir', type=str, default='modelFood/log/')
parser.add_argument('--save_dir', type=str, default='modelFood/save/')
parser.add_argument('--data_dir', type=str, default='foodData/')
parser.add_argument('--submission_dir', type=str, default='submission/')
parser.add_argument('--submit', default=True)
parser.add_argument('--analysis', default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_iter', type=int, default=100)
parser.add_argument('--n_batch', type=int, default=8)
parser.add_argument('--max_grad_norm', type=int, default=1)
parser.add_argument('--lr', type=float, default=6.25e-5)
parser.add_argument('--lr_warmup', type=float, default=0.002)
parser.add_argument('--n_ctx', type=int, default=512)
parser.add_argument('--n_embd', type=int, default=32)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--clf_pdrop', type=float, default=0.1)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--vector_l2', action='store_true')
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--afn', type=str, default='gelu')
parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
parser.add_argument('--encoder_path', type=str, default='modelFood/model/encoder_food_20.json')
parser.add_argument('--bpe_path', type=str, default='modelFood/model/vocab_40000.bpe')
parser.add_argument('--n_transfer', type=int, default=12)
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument('--b1', type=float, default=0.9)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--e', type=float, default=1e-8)


class ModelFood(BaseModel):



    def add_placeholders(self):


        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        self.char_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="char_ids")

        self.angle1s = tf.placeholder(tf.float32, shape=[None, None],
                                       name="angle1s")

        self.angle2s = tf.placeholder(tf.float32, shape=[None, None],
                                      name="angle2s")
        self.label_ids=tf.placeholder(tf.int32, shape=[None, None],
                        name="label_ids")

        self.sp_token_weight = tf.placeholder(tf.float32, shape=[None, None],
                                        name="sp_token")



        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

    # line_c, removed.
    def get_feed_dict(self, chars, angle1s ,angle2s,labels,line_c,sp_token,lr=None, dropout=None):


        char_ids, sequence_lengths = pad_sequences(chars, pad_tok=0)
        label_ids, sequence_lengths = pad_sequences(labels, pad_tok=0)
        sp_token, sequence_lengths = pad_sequences(sp_token, pad_tok=0)

        feed = {
            self.char_ids: char_ids,
            self.sequence_lengths: sequence_lengths,
            self.label_ids: label_ids,#1,2,3,4


            self.sp_token_weight:sp_token

        }
        # print(feed)

        if angle1s is not None:
            angle1s, _ = pad_sequences(angle1s, pad_tok=0)
            feed[self.angle1s] = angle1s
        if angle2s is not None:
            angle2s, _ = pad_sequences(angle2s, pad_tok=0)
            feed[self.angle2s] = angle2s

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def add_word_embeddings_op(self):

        if(self.config.use_transformer==False):

            _char_embeddings = tf.get_variable(
                name="_char_embeddings",
                dtype=tf.float32,
                shape=[self.config.nchars, self.config.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                     self.char_ids, name="char_embeddings")


            cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_char,
                                              state_is_tuple=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_char,
                                              state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length= self.sequence_lengths, dtype=tf.float32,scope='rnn1')

            (output_fw, output_bw),_ = _output

            output_concate = tf.concat([output_fw, output_bw], axis=-1)


            char_embeddings = output_concate


            if(self.config.ngram_embed==True):

                kernels = tf.get_variable("kernels", shape=[self.config.ngram,2*self.config.hidden_size_char,self.config.n_kernel],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

                char_embeddings = tf.nn.conv1d(char_embeddings, kernels, stride=1, padding="SAME")

                char_embeddings = tf.nn.relu(char_embeddings)
                #print("charemb",char_embeddings)


        self.M3_fw = tf.get_variable("M3_fw", dtype=tf.float32,
                                  shape=[self.config.hidden_size_char , self.config.label_emb_size],initializer=tf.contrib.layers.xavier_initializer())


        self.M3_bw = tf.get_variable("M3_bw", dtype=tf.float32,
                                     shape=[self.config.hidden_size_char, self.config.label_emb_size],initializer=tf.contrib.layers.xavier_initializer())


        self.M4k_fw = tf.get_variable("M4k_fw", dtype=tf.float32,
                                     shape=[(self.config.use_K_histroy+1)*self.config.label_emb_size,
                                            self.config.hidden_size_char],initializer=tf.contrib.layers.xavier_initializer())


        self.M4k_bw = tf.get_variable("M4k_bw", dtype=tf.float32,
                                     shape=[(self.config.use_K_histroy+1)*self.config.label_emb_size ,
                                            self.config.hidden_size_char],initializer=tf.contrib.layers.xavier_initializer())


        if(self.config.use_label_linkM4==True):

            cell_fw2 = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                               state_is_tuple=True)
            cell_bw2 = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                               state_is_tuple=True)
            if(self.config.use_attention==True):
                #print("M4 attention")

                cell_fw2 = NlabelCell.NLabelAttentionCellWrapper(cell_fw2, self.config.ngram, emb_M3=self.M3_fw,
                                                                emb_M4k=self.M4k_fw, state_is_tuple=True) #attn len is not used
                cell_bw2 = NlabelCell.NLabelAttentionCellWrapper(cell_bw2, self.config.ngram, emb_M3=self.M3_bw,
                                                                emb_M4k=self.M4k_bw, state_is_tuple=True)
            else:
                #print("M4 no attention")
                cell_fw2 = NlabelCell.NLabelNoAttentionCellWrapper(cell_fw2, emb_M3=self.M3_fw,
                                                               emb_M4k=self.M4k_fw, state_is_tuple=True)
                cell_bw2 = NlabelCell.NLabelNoAttentionCellWrapper(cell_bw2, emb_M3=self.M3_bw,
                                                               emb_M4k=self.M4k_bw, state_is_tuple=True)


        else:

            cell_fw2 = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                               state_is_tuple=True)
            cell_bw2 = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                                               state_is_tuple=True)

            #print("use simple LSTM")




        _output = tf.nn.bidirectional_dynamic_rnn(
            cell_fw2, cell_bw2, char_embeddings,
            sequence_length=self.sequence_lengths, dtype=tf.float32,scope='rnn_output')



        (output_fw, output_bw), _ = _output

        output_concate = tf.concat([output_fw, output_bw], axis=-1)


        char_embeddings = output_concate

        #print("char emb out",char_embeddings)


        self.char_embedding=tf.nn.dropout(char_embeddings, self.dropout)
        self.fwbw=(output_fw, output_bw)


    def add_logits_op(self):



        fw,bw=self.fwbw

        fw_shape=tf.shape(fw)

        bw_shape=tf.shape(bw)

        fw=tf.reshape(fw, shape=[-1,self.config.hidden_size_char])

        bw = tf.reshape(bw, shape=[-1, self.config.hidden_size_char])

        pre_logits_labels_fw = tf.nn.relu(tf.matmul(fw, self.M3_fw))

        pre_logits_labels_bw = tf.nn.relu(tf.matmul(bw, self.M3_bw))

        pre_logits_labels_fw=tf.reshape(pre_logits_labels_fw ,shape= [fw_shape[0],fw_shape[1],self.config.label_emb_size ])

        pre_logits_labels_bw = tf.reshape(pre_logits_labels_bw, shape=[bw_shape[0],bw_shape[1],self.config.label_emb_size ])

        pre_logits_labels_concat=tf.concat([pre_logits_labels_fw, pre_logits_labels_bw],axis=-1)

        self.logits_labels=tf.layers.dense(inputs=pre_logits_labels_concat,units=self.config.nlabels,activation=tf.nn.relu)


        self.logits = tf.layers.dense(inputs=self.char_embedding, units=self.config.output_dim)


    def add_loss_op(self):

        if self.config.loss_op=="msn":
            if(self.config.use_triangle==True):
                self.angles_stacked = tf.stack(
                    [tf.sin(self.angle1s / 180 * math.pi), tf.cos(self.angle1s / 180 * math.pi),
                     tf.sin(self.angle2s / 180 * math.pi), tf.cos(self.angle2s / 180 * math.pi)], axis=2)
            else:

                self.angles_stacked=tf.stack([self.angle1s, self.angle2s],axis=2)


            label_ids=tf.one_hot(indices=self.label_ids,depth=self.config.nlabels)

            self.crossEntropy=tf.nn.softmax_cross_entropy_with_logits( logits=self.logits_labels,labels= label_ids) #output : element-wised

            if (self.config.weighted_loss != 1):
                self.crossEntropy = tf.multiply(self.crossEntropy, self.sp_token_weight)
                self.angles_stacked = tf.multiply(self.angles_stacked, tf.expand_dims(self.sp_token_weight,-1))


            self.zero_mask = tf.where(tf.not_equal(self.char_ids, 0))


            self.masked_angles_stacked = tf.gather_nd(self.angles_stacked,self.zero_mask)


            if(self.config.use_triangle==True):
                self.logger.info("using l2 norm")
                s_logits=tf.shape(self.logits)
                tmp_logits=tf.reshape(self.logits,shape=[s_logits[0],s_logits[1],2,2])
                tmp_logits=tf.nn.l2_normalize(tmp_logits,dim=-1)
                self.logits=tf.reshape(tmp_logits,shape=[s_logits[0],s_logits[1],4])


            if (self.config.weighted_loss !=1):
                self.logits = tf.multiply(self.logits, tf.expand_dims(self.sp_token_weight,-1))


            self.masked_logits = tf.gather_nd(self.logits, self.zero_mask)

            self.masked_crossEntropy=tf.gather_nd(self.crossEntropy,self.zero_mask)



            self.loss_angle = tf.losses.mean_squared_error(self.masked_angles_stacked, self.masked_logits) # +self.masked_crossEntropy


            self.loss_label=tf.reduce_mean(self.masked_crossEntropy)
            self.loss_label=self.loss_label/8

            if(self.config.use_label_info==True):
                self.loss=self.loss_angle +self.loss_label
            else:
                self.loss=self.loss_angle

        tf.summary.scalar("loss", self.loss)

    def build(self):
        with tf.variable_scope("mergedModel"):
            self.add_placeholders()
            self.add_word_embeddings_op()
            self.add_logits_op()
            self.add_loss_op()

            self.add_train_op(self.config.lr_method, self.lr, self.loss,
                              self.config.clip)



            self.initialize_session()


            print("finish build")



    def predict_batch(self, words):

        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

        return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):

        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        # iterate over dataset
        for i, (chars, angle1s,angle2s,labels,line_c, sp_token) in enumerate(minibatches(train, batch_size)):



            fd, _ = self.get_feed_dict(chars, angle1s,angle2s,labels,line_c,sp_token, self.config.lr,
                                       self.config.dropout)



            _, train_loss, summary,ce,ag,m_ce,m_l = self.sess.run([self.train_op, self.loss, self.merged,self.loss_label,self.loss_angle,self.masked_crossEntropy,self.masked_logits], feed_dict=fd)

            print("current batch loss :")
            print("ce",ce)
            print("ag",ag)

            prog.update(i + 1, [("train loss", train_loss)])

            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["loss"]

    def run_evaluate(self, test):

        batch_size = self.config.batch_size
        sum_loss=0
        for i, (chars, angle1s, angle2s,labels,line_c,sp_token) in enumerate(minibatches(test, batch_size)):
            fd, _ = self.get_feed_dict(chars, angle1s, angle2s,labels,line_c,sp_token, lr=0,dropout=1)

            test_loss= self.sess.run(
                [self.loss], feed_dict=fd)

            sum_loss=sum_loss+test_loss[0]

        return {"loss": sum_loss/(i+1),"average of validation batches num:": (i+1) }


    def predict(self, data):

        batch_size = self.config.batch_size
        preds=[]
        ang1=[]
        ang2=[]

        for i, (chars, angle1s, angle2s,labels,line_c,sp_token) in enumerate(minibatches(data, batch_size)):
            if(i%1000==0):
                print("test at batch:",i)

            fd, _ = self.get_feed_dict(chars, angle1s, angle2s,labels, line_c,sp_token, lr=None,dropout=1)

            p= self.sess.run(
                [self.logits,self.angle1s,self.angle2s,self.label_ids], feed_dict=fd)

            preds.append(p[0])
            ang1.append(p[1])
            ang2.append(p[2])

        preds=np.asarray(preds)
        ang1 = np.asarray(ang1)
        ang2 = np.asarray(ang2)

        return preds,ang1,ang2

