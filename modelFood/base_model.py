import os
import tensorflow as tf
from .config import Config


class BaseModel(object):

    def __init__(self, config):

        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None
        self.config = Config()
        self.modelName="seqLen_"+str(self.config.seqLen)+"_ngram_embed_"+str(self.config.ngram_embed)+"ngram_"+str(self.config.ngram) +"use_tri_"+\
                       str(self.config.use_triangle)+"use_label_info"+str(self.config.use_label_info)+"use_kmean"+str(self.config.use_label_kmean)+\
                       "use_attention_"+str(self.config.use_attention)+"use_M4"+str(self.config.use_label_linkM4)+ "k_histroy_"+str(self.config.use_K_histroy)+"transformer"+str(self.config.use_transformer)+"_weightedLoss_"+str(self.config.weighted_loss)

        #self.modelName="debugAtt"
        #print("modelName",self.modelName)

    def reinitialize_weights(self, scope_name):

        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1):

        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def initialize_session(self):

        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #self.sess=sess

        self.saver = tf.train.Saver(max_to_keep=3)



    def restore_session(self, dir_model):
        print("restore:",dir_model)

        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)


    def save_session(self,i):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model+self.modelName,global_step=i)


    def close_session(self):
        """Closes the session"""
        self.sess.close()


    def add_summary(self):

        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)


    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        """
        best_score = 1000000
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary() # tensorboard

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            self.config.lr *= self.config.lr_decay # decay learning rate
            # early stopping and saving best parameters
            if score <= best_score:
                nepoch_no_imprv = 0
                self.save_session(epoch)
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break


    def evaluate(self, test):
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")

        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)
