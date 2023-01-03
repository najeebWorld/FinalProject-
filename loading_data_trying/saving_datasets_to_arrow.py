import json

import torch
import torchaudio
from datasets import load_dataset, Features, Value, Audio
import re
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

print("----------------- Checking if cuda is available... -----------------")
print(f'Cuda Available = {torch.cuda.is_available()}\n\n')

print("----------------- Loading Datasets... -----------------")
train = load_dataset("mozilla-foundation/common_voice_11_0", "pt", use_auth_token=True)
train.save_to_disk('/home/or/Desktop/portu')
# test = load_dataset("mozilla-foundation/common_voice_11_0", "tr", split="validation", use_auth_token=True)
print("----------------- Loading Datasets complete. -----------------\n\n")

# CHARS_TO_IGNORE = ["r", "o", "f", "n", "d", "c", "y", "m", "t", "h", "x", "l", "u", "i", "w", "s", "a", ",", "?", "¿",
#                    ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
#                    "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
#                    "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
#                    "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
#                    "『", "』", "〝", "〟", "⟨", "⟩", "〜", '#', 'get', '  ', '\'ۖ ', "：", "！", '؟', '\'ۚ', "？", "♪", "؛",
#                    '%', "/", '\'', '☭', "\\", "º", '؛', "−", 'َ', "^", 'ّ', "'", 'ً', "ʻ", 'ٍ', "ˆ", 'ِ', 'ُ', 'ٓ', 'ٰ',
#                    'ْ', 'ٌ', '31', '24', '39']
#
# chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '',
                               batch["sentence"]).lower()  # remove " " for the inference split
    return batch


print("----------------- Removing special characters... -----------------")
train = train.map(remove_special_characters)
validation = validation.map(remove_special_characters)
print("----------------- Removing special characters complete. -----------\n\n")


def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


print("----------------- Extracting all characters... -----------------")
vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                        remove_columns=train.column_names)
vocab_test = validation.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True,
                            remove_columns=validation.column_names)
print("----------------- Extracting all characters complete. -----------------\n\n")

# ----------------------------------- VOCAB -----------------------------------#

print("----------------- Preparing vocab... -----------------")
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

print(f'Vocab_dict: {vocab_dict}')

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(f'Vocab_len: {len(vocab_dict)}')

print("----------------- Preparing vocab complete. -----------------\n\n")

print("----------------- Saving vocab to jason... -----------------")
with open('vocab_no_eng.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

print("----------------- Saving vocab to jason complete. -----------------\n\n")

tokenizer = Wav2Vec2CTCTokenizer("/home/or/Desktop/wav2vec2/loading_data_trying/vocab_no_eng.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# ----------------------------------- Preparing datasets -----------------------------------#

def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


print("----------------- Preparing datasets... -----------------")
split = split.map(prepare_dataset, remove_columns=split.column_names)  # maybe we'll have to reduce to 1
print("\n\n----------------- Preparing datasets complete. -----------------\n\n")

# That for the test split for inference
# resampler = torchaudio.transforms.Resample(48_000, 16_000)
#
#
# # Preprocessing the datasets.
# # We need to read the audio files as arrays
# def speech_file_to_array_fn(batch):
#     speech_array, sampling_rate = torchaudio.load(batch['path'])
#     batch["speech"] = resampler(speech_array).squeeze().numpy()
#     return batch
#
#
# print("----------------- Preparing datasets... -----------------")
# split = split.map(speech_file_to_array_fn, drop_last_batch=True)
# print("\n--------------- Preparing datasets complete. -----------\n\n")

print("----------------- Saving dataset to arrow file... -----------------")
split.save_to_disk('/home/or/Desktop/arabic_new_dataset/arrow/test')
print("----------------- Saving dataset to arrow file complete. ----------\n")
