import numpy as np
from sklearn.utils import shuffle
from .kmeanInfer import kmeanInfer

with open("modelFood/modelSettings.txt") as f:
    modelSettings = f.readlines()
    # print(modelSettings)

use_label_kmean = False if (modelSettings[3].strip().split(":")[1] == "False") else True

seqLen = int(modelSettings[7].strip().split(":")[1])
weighted_loss = float(modelSettings[8].strip().split(":")[1])

# special error message
class MyIOError(Exception):
    def __init__(self, filename):

        # custom error message
        message = """
        
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)

class FoodDataset(object):
    def __init__(self, filename,processing_word=None,processing_label=None,
                 max_iter=None, Phase="build"):

        self.filename = filename.strip()
        self.processing_word = processing_word
        self.processing_label=processing_label
        self.max_iter = max_iter
        self.length = None
        self.Phase=Phase


        self.src_char =np.load(self.filename.split(" ")[0])
        self.src_angle1 = np.load(self.filename.split(" ")[1])
        self.src_angle2 = np.load(self.filename.split(" ")[2])
        self.src_angleLabel = np.load(self.filename.split(" ")[3])
        self.use_label_kmean=use_label_kmean
        self.seqLen=seqLen


        if (self.seqLen == 200):
            self.trainLen = 20000
        if (self.seqLen == 350):
            self.trainLen = 45000
        if (self.seqLen == 700):
            self.trainLen = 75000

        f.close()


    def __iter__(self):
        niter = 0


        if(self.Phase=="train"):
            src_char = self.src_char[:self.trainLen]
            src_angle1 = self.src_angle1[:self.trainLen]
            src_angle2 = self.src_angle2[:self.trainLen]
            src_angleLabel=self.src_angleLabel[:self.trainLen]

            src_char, src_angle1, src_angle2, src_angleLabel = shuffle(src_char, src_angle1, src_angle2,src_angleLabel)

        elif(self.Phase=="val"):
            src_char = self.src_char[self.trainLen:]
            src_angle1 = self.src_angle1[self.trainLen:]
            src_angle2 = self.src_angle2[self.trainLen:]
            src_angleLabel = self.src_angleLabel[self.trainLen:]

            #src_char, src_angle1, src_angle2 = shuffle(src_char, src_angle1, src_angle2)

        elif(self.Phase=="infer"):
            src_char=np.load("foodData/doinfer.npy")
            src_angle1=np.load("foodData/doinfer_ang1_tgt.npy")
            src_angle2= np.load("foodData/doinfer_ang2_tgt.npy")
            src_angleLabel = np.load("foodData/doinfer_label.npy")

        elif (self.Phase == "infer_notgt"):
            src_char=np.load("/Users/ibm_siyuhuo/GithubRepo/seq2seq/foodData/doinfer.npy")


        else:
            src_char = self.src_char
            src_angle1 = self.src_angle1
            src_angle2 = self.src_angle2
            src_angleLabel = self.src_angleLabel

        for i in range(len(src_char)):
            niter=niter+1
            line=src_char[i]
            line_label=src_angleLabel[i]
            line_processed = self.processing_word(line)
            line_label_processed=self.processing_label(line_label)
            #print("line_label_processed",line_label_processed)

            if(self.Phase=='infer_notgt'):
               #
                angle1=[0]*len(line)
                angle2=[0]*len(line)
                # print(angle2)
            else:
                angle1=list(np.float_(src_angle1[i]))
                angle2=list(np.float_(src_angle2[i]))

                angle=list(zip(angle1,angle2))
                line_label_Kmean=np.asarray(kmeanInfer(angle))

                #print("kmean",line_label_Kmean)
                # print("process",np.asarray(line_label_processed))

            if(self.use_label_kmean==True):
                #print("yield kmean")
                #print(line_label_Kmean)
                yield line_processed,angle1,angle2,line_label_Kmean," ".join(line),line_label_processed
            else:
                #pr print(line_label_processed)int("yield processed label")
                #print(line_label_processed)
                yield line_processed, angle1, angle2, line_label_processed," ".join(line),line_label_processed

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length



def get_char_vocab(dataset):

    vocab_char = set()
    for line,angle1,angle2,angleLabel in dataset:

            vocab_char.update(line)

    return vocab_char


def get_label_vocab(dataset):
    vocab_label = set()
    for line, angle1, angle2, angleLabel in dataset:
        vocab_label.update(angleLabel)

    return vocab_label

def write_vocab(vocab, filename):

    print("Writing vocab...")
    with open(filename, "w") as f:
        f.write("{}\n".format('$'))
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(str(word))
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def get_processing_word(vocab_chars=None):
    def f(word):
        if(len(vocab_chars)==0):
            char_items = []
            for char in word:
                char_items.append(char)
            return char_items

        if vocab_chars is not None:
            char_ids = []
            for char in word:
                if char in vocab_chars:
                    # print(len(vocab_chars))
                    # print(char)
                    # print(vocab_chars[char])
                    char_ids += [vocab_chars[char]]

            return char_ids
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):

   
    #max_length = max(map(lambda x : len(x), sequences))
    max_length =seqLen
    sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)
    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):

    x_batch, y1_batch,y2_batch, l_batch,c_batch,special_token_batch = [], [], [], [], [],[]
    for (x, y1,y2, l,c,org_l) in data: #org_l is copy for original label
        # print("x",x)
        # print("y1",y1)
        # print("y2",y2)
        # print("l",l)
        # print("c",c)
        # print("org_l",org_l)
        if len(x_batch) == minibatch_size:
            yield x_batch, y1_batch, y2_batch, l_batch, c_batch, special_token_batch
            x_batch, y1_batch, y2_batch, l_batch, c_batch, special_token_batch = [], [], [], [], [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y1_batch += [y1]
        y2_batch += [y2]
        l_batch += [l]
        c_batch += [c]
        tmp=[]
        for t in org_l:
            if(t==3 ):
                tmp.append(weighted_loss)
            else:
                tmp.append(1)
        special_token_batch += [tmp]


    if len(x_batch) != 0:
        yield x_batch, y1_batch, y2_batch, l_batch, c_batch, special_token_batch


def get_chunk_type(tok, idx_to_tag):

    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):

    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
