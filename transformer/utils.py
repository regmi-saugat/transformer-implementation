from logging import getLogger
from datasets import load_dataset
from transformer.train import get_or_build_tokenizers
from torch.utils.data import Dataset, DataLoader, random_split


from dataset import BilingualDataset, casual_mask

logger = getLogger()


class Utils:
    @staticmethod
    def get_all_sentences(dataset, language):
        for item in dataset:
            yield item["translation"][language]

    @staticmethod
    def get_datasets(config):
        # Only has train split, so we divide it overselves
        raw_data = load_dataset(
            f"{config['datasource']}",
            f"{config['language_source']}-{config['language_target']}",
            split="train",
        )

        # Build tokenizers
        tokenizer_source = get_or_build_tokenizers(
            config, raw_data, config["language_source"]
        )

        tokenizer_target = get_or_build_tokenizers(
            config, raw_data, config["language_target"]
        )

        # Keep 90% for training, and 10% for validation
        train_dataset_size = int(0.9 * len(raw_data))
        validation_dataset_size = len(raw_data) - train_dataset_size
        train_dataset_raw, validation_dataset_raw = random_split(
            raw_data, [train_dataset_size, validation_dataset_size]
        )

        train_data = BilingualDataset(
            train_dataset_raw,
            tokenizer_source,
            tokenizer_target,
            config["language_source"],
            config["language_target"],
            config["seq_len"],
        )
        validation_data = BilingualDataset(
            validation_dataset_raw,
            tokenizer_source,
            tokenizer_target,
            config["language_source"],
            config["language_target"],
            config["seq_len"],
        )

        # Find maximum length of each sentence in the source and target sentence
        max_len_source = 0
        max_len_target = 0

        for item in raw_data:
            source_ids = tokenizer_source.encode(
                item["translation"][config["language_source"]]
            ).ids
            target_ids = tokenizer_target.encode(
                item["translation"][config["language_target"]]
            ).ids
            max_len_source = max(max_len_source, len(source_ids))
            max_len_target = max(max_len_target, len(target_ids))

        logger.info(f"Maximum length of source sentence: {max_len_source}")
        logger.info(f"Maximu length of target sentence: {max_len_target}")

        train_dataloader = DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True
        )
        validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)

        return train_dataloader, validation_dataloader
