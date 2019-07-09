import sys
from modelFood.config import Config
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
config = Config()
np.set_printoptions(threshold=np.nan)
sys.stdout = open('./checkSave1.txt', 'w')
print_tensors_in_checkpoint_file(file_name=config.dir_model+'debugAtt-1', tensor_name='', all_tensors=True, all_tensor_names=True)
