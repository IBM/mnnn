from modelFood.data_utils import FoodDataset
from modelFood.model_food import ModelFood
from modelFood.config import Config
import numpy as np
import datetime


def main():
    # create instance of config
    config = Config()

    model = ModelFood(config)
    model.build()
    test   = FoodDataset(config.filename_whole, config.processing_word,config.processing_labels, config.max_iter,Phase="infer")

    model.restore_session(config.dir_model+"seqLen_200_ngram_embed_Truengram_21use_tri_Trueuse_label_infoTrueuse_kmeanTrueuse_attention_Falseuse_M4Truek_histroy_21transformerFalse_weightedLoss_2.0_update1-63")
    #model.restore_session("/Users/ibm_siyuhuo/Desktop/rnncnnrnn_best_ccc")
    #print(model.run_evaluate(test))

    st=datetime.datetime.now()

    preds, ang1, ang2=model.predict(test)
    ed=datetime.datetime.now()
    print("usetime",ed-st)

    # res=np.array(res)
    # print(res)
    np.save("results/npyRes/preds.npy",preds)
    np.save("results/npyRes/ang1.npy", ang1)
    np.save("results/npyRes/ang2.npy", ang2)

if __name__ == "__main__":
    main()


