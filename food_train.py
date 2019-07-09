
from modelFood.data_utils import FoodDataset
from modelFood.model_food import ModelFood
from modelFood.config import Config

def main():
    # create instance of config
    config = Config()

    model = ModelFood(config)
    model.build()
    dev   = FoodDataset(config.filename_whole, config.processing_word,config.processing_labels, config.max_iter,Phase="val")
    train = FoodDataset(config.filename_whole, config.processing_word,config.processing_labels, config.max_iter,Phase="train")
    model.train(train, dev)

if __name__ == "__main__":
    main()
