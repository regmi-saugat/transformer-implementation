import torch

# import torchmetrics
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from dataset import BilingualDataset, causal_mask
from transformer.train import get_or_build_tokenizers

from logging import getLogger

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
            f"{config['lang_src']}-{config['lang_tgt']}",
            split="train",
        )

        # Build tokenizers
        tokenizer_source = get_or_build_tokenizers(config, raw_data, config["lang_src"])

        tokenizer_target = get_or_build_tokenizers(config, raw_data, config["lang_tgt"])

        # Keep 90% for training, and 10% for validation
        train_dataset_size = int(0.9 * len(raw_data))
        validation_dataset_size = len(raw_data) - train_dataset_size
        train_dataset_raw, validation_dataset_raw = random_split(
            raw_data, [train_dataset_size, validation_dataset_size]
        )
        # Creating a data that model use
        train_data = BilingualDataset(
            train_dataset_raw,
            tokenizer_source,
            tokenizer_target,
            config["lang_src"],
            config["lang_tgt"],
            config["seq_len"],
        )
        validation_data = BilingualDataset(
            validation_dataset_raw,
            tokenizer_source,
            tokenizer_target,
            config["lang_src"],
            config["lang_tgt"],
            config["seq_len"],
        )

        # Find maximum length of each sentence in the source and target sentence
        max_len_source = 0
        max_len_target = 0

        for item in raw_data:
            source_ids = tokenizer_source.encode(
                item["translation"][config["lang_src"]]
            ).ids
            target_ids = tokenizer_target.encode(
                item["translation"][config["lang_tgt"]]
            ).ids
            max_len_source = max(max_len_source, len(source_ids))
            max_len_target = max(max_len_target, len(target_ids))

        logger.info(f"Maximum length of source sentence: {max_len_source}")
        logger.info(f"Maximu length of target sentence: {max_len_target}")

        train_dataloader = DataLoader(
            train_data, batch_size=config["batch_size"], shuffle=True
        )
        validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)

        return (
            train_dataloader,
            validation_dataloader,
            tokenizer_source,
            tokenizer_target,
        )

    @staticmethod
    def run_validation(
        model,
        validation_ds,
        tokenizer_src,
        tokenizer_tgt,
        max_len,
        device,
        print_msg,
        global_step,
        writer,
        num_examples=2,
    ):
        model.eval()
        count = 0

        # source_texts = []
        # expected = []
        # predicted = []

        # size of control window (using default value)
        console_width = 80

        with torch.no_grad():
            for batch in validation_ds:
                count = +1
                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"
                model_output = Utils.greedy_decode(
                    model,
                    encoder_input,
                    encoder_mask,
                    tokenizer_src,
                    tokenizer_tgt,
                    max_len,
                    device,
                )
                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_output_text = tokenizer_tgt.decode(
                    model_output.detach().cpu().numpy()
                )

                # source_texts.append(source_text)
                # expected.append(target_text)
                # predicted.append(model_output_text)

                # Print the source, target, and model output
                print_msg("-" * console_width)
                print_msg(f"{'SOURCE: ':>12}{source_text}")
                print_msg(f"{'TARGET: ':>12}{target_text}")
                print_msg(f"{'PREDICTED: ':>12}{model_output_text}")

                if count == num_examples:
                    print_msg("-" * console_width)
                    break
        # if writer:
        #     # Evaluate the character error rate
        #     # Compute the char error rate
        #     metric = torchmetrics.CharErrorRate()
        #     cer = metric(predicted, expected)
        #     writer.add_scalar('validation cer', cer, global_step)
        #     writer.flush()

        #     # Compute the word error rate
        #     metric = torchmetrics.WordErrorRate()
        #     wer = metric(predicted, expected)
        #     writer.add_scalar('validation wer', wer, global_step)
        #     writer.flush()

        #     # Compute the BLEU metric
        #     metric = torchmetrics.BLEUScore()
        #     bleu = metric(predicted, expected)
        #     writer.add_scalar('validation BLEU', bleu, global_step)
        #     writer.flush()

    @staticmethod
    def greedy_decode(
        model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
    ):
        sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = tokenizer_tgt.token_to_id("[EOS]")

        # Precompute encoder output and reuse it for every token we get from encoder
        encoder_output = model.encode(source, source_mask)
        # Initialize the decoder output with the sos token
        decoder_input = (
            torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device=device)
        )
        while True:
            if decoder_input.size(1) == max_len:
                break
            # Build mask for the target (decoder input)
            decoder_mask = (
                causal_mask(decoder_input.size(1))
                .type_as(source_mask)
                .to(device=device)
            )

            # Calculate the output of the decoder
            output = model.decode(
                encoder_output, source_mask, decoder_input, decoder_mask
            )

            # Get the mask token
            probab = model.projection(output[:, -1])
            _, next_word = torch.max(probab, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1)
                    .type_as(source)
                    .fill_(next_word.item())
                    .to(device=device),
                ],
                dim=1,
            )

            if next_word == eos_idx:
                break
        return decoder_input.squeeze(0)
