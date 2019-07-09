import os
from .general_utils import get_logger
from .data_utils import load_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.logger = get_logger(self.path_log)

        if load:
            self.load()

    def load(self):

        with open("modelFood/modelSettings.txt") as f:
            self.modelSettings= f.readlines()

        #print(modelSettings)
        self.use_triangle = False if(self.modelSettings[0].strip().split(":")[1]=="False") else True
        #print("use_triangle",self.use_triangle)
        self.use_attention = False if(self.modelSettings[1].strip().split(":")[1]=="False") else True
        #print("use_attention",self.use_attention)

        self.use_label_info = False if(self.modelSettings[2].strip().split(":")[1]=="False") else True
        #print("use_label_info",self.use_label_info)

        self.use_label_kmean = False if(self.modelSettings[3].strip().split(":")[1]=="False") else True
        #print("use_label_Kmean",self.use_label_kmean)

        self.ngram_embed = False if(self.modelSettings[4].strip().split(":")[1]=="False") else True
        #print("ngram_embed",self.ngram_embed)

        self.use_label_linkM4 = False if (self.modelSettings[5].strip().split(":")[1] == "False") else True
        #print("ngram_embed", self.use_label_linkM4)

        self.use_transformer= False if (self.modelSettings[6].strip().split(":")[1] == "False") else True
        #print("use_transformer", self.use_transformer)

        self.seqLen= int(self.modelSettings[7].strip().split(":")[1])
        #print("seqLen", self.seqLen)

        self.weighted_loss = float(self.modelSettings[8].strip().split(":")[1])
        #print("weightedLoss", self.weighted_loss)

        self.vocab_chars = load_vocab(self.filename_chars)

        self.vocab_labels=load_vocab(self.filename_labels)

        self.nchars = len(self.vocab_chars)

        if(self.use_label_kmean==False):

            self.nlabels=len(self.vocab_labels)
            #print("processed nlabels",self.nlabels)
        else:
            self.nlabels=101

        if(self.use_triangle==True):
            self.output_dim=4  #angle1: cos, sin  angle2: cos , sin
        else:
            self.output_dim=2



        if (self.seqLen == 200):
            self.trainLen = 20000
            self.filename_src_char = 'foodData/src_angle_200.npy'

            # filename_src_char_200_9gram='foodData/src_angle_200_9gram_central_char.npy'

            self.filename_src_angle1 = 'foodData/tgt_angle1_200.npy'

            self.filename_src_angle2 = "foodData/tgt_angle2_200.npy"

            self.filename_src_angleLabel = "foodData/src_angleLabel_200.npy"

        if(self.seqLen==350):
            self.trainLen= 45000
            self.filename_src_char = 'foodData/src_angle_350.npy'

            # filename_src_char_200_9gram='foodData/src_angle_200_9gram_central_char.npy'

            self.filename_src_angle1 = 'foodData/tgt_angle1_350.npy'

            self.filename_src_angle2 = "foodData/tgt_angle2_350.npy"

            self.filename_src_angleLabel = "foodData/src_angleLabel_350.npy"

        if(self.seqLen==700):
            self.trainLen= 75000
            self.filename_src_char = 'foodData/src_angle_700.npy'

            # filename_src_char_200_9gram='foodData/src_angle_200_9gram_central_char.npy'

            self.filename_src_angle1 = 'foodData/tgt_angle1_700.npy'

            self.filename_src_angle2 = "foodData/tgt_angle2_700.npy"

            self.filename_src_angleLabel = "foodData/src_angleLabel_700.npy"


        self.filename_whole = self.filename_src_char + " " + self.filename_src_angle1 + " " + self.filename_src_angle2 + " " + self.filename_src_angleLabel



        self.processing_word = get_processing_word(self.vocab_chars)

        self.processing_labels= get_processing_word(self.vocab_labels)

        self.loss_op="msn"


        f.close()

    # general config
    dir_output = "results/test/"
    
    dir_model  = dir_output + "model_food.weights/"
    
    path_log   = dir_output + "log_food.txt"

    dim_char = 16

    ngram = 21

    use_K_histroy= 21


    n_kernel=16

    n_centers=64

    dense_out_dim=5

    use_pretrained = False


    label_emb_size = 32
    hidden_size_char = 32  # lstm on chars, equal to label emb size

    q_len=7


    filename_infer="foodData/doinfer.npy"


    max_iter = None # if not None, max number of examples in Dataset

    filename_chars = "foodData/food_chars.txt"
    filename_labels="foodData/food_labels.txt"
    # training
    train_embeddings = False
    nepochs          = 100
    dropout          = 0.8
    batch_size       = 8
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 20




