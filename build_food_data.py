from modelFood.config import Config
from modelFood.data_utils import FoodDataset,write_vocab, get_char_vocab,get_label_vocab


def main():

    # get config and processing of words
    config = Config()
    data = FoodDataset(config.filename_whole, config.processing_word,config.processing_labels, config.max_iter,Phase="build")
    vocab_chars = get_char_vocab(data)
    vocab_labels= get_label_vocab(data)

    write_vocab(vocab_chars, config.filename_chars)
    write_vocab(vocab_labels,config.filename_labels)


if __name__ == "__main__":
    main()
