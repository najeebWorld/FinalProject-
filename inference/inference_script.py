import json

import torch
import torchaudio
from datasets import load_dataset, Audio, load_from_disk, Features, Value
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from datasets import load_metric
import re

wer = load_metric("wer")
cer = load_metric("cer")

# LANG_ID = "ar"
# MODEL_ID = "../20_gaussian/checkpoint_2104"
# DATASET = "mozilla-foundation/common_voice_11_0"
#
# print('---------------- Loading Data... ---------------------')
# test_dataset = load_dataset(DATASET, LANG_ID, split="test")
# test_dataset = test_dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
# print('---------------- Loading Data complete. ---------------------\n\n')


features = Features(
    {
        "client_id": Value("string"),
        "path": Value("string"),
        "audio": Audio(sampling_rate=48_000),
        "sentence": Value("string"),
        "up_votes": Value("int64"),
        "down_votes": Value("int64"),
        "age": Value("string"),
        "gender": Value("string"),
        "accents": Value("string"),
        "locale": Value("string"),
        "segment": Value("string"),
    }
)

print("----------------- Loading Datasets... -----------------")
sample_data = load_dataset('csv', data_files={'test': 'test.csv', }, data_dir='/home/or/Desktop/turkish')
print("----------------- Loading Datasets complete. ----------\n\n")

print("----------------- Casting features... -----------------")
sample_data = sample_data.cast(features)
print("----------------- Casting features complete. -----------\n\n")

print("----------------- Removing columns... -----------------")
sample_data = sample_data.remove_columns(
    ["accents", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
print("----------------- Removing columns complete. -----------\n\n")

print("----------------- Loading audio from path... -----------------")
test_dataset = sample_data['test']
test_dataset['audio']
print("----------------- Loading audio from path complete. -----------\n\n")
#
#
# CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
#                    "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
#                    "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
#                    "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
#                    "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ"]
#
# chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch


print("----------------- Removing special characters... -----------------")
test_dataset = test_dataset.map(remove_special_characters)

print('---------------- Loading processor and model... ---------------------')

tokenizer = Wav2Vec2CTCTokenizer("../vocab_turkish_aug.json", unk_token="[UNK]",
                                 pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model = Wav2Vec2ForCTC.from_pretrained('../turkish_augmented/checkpoint-20692')
print('---------------- Loading processor and model complete. ---------------------\n\n')
#
resampler = torchaudio.transforms.Resample(48_000, 16_000)


# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])

    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch


print('---------------- Processing the data... ---------------------')
test_dataset = test_dataset.map(speech_file_to_array_fn)
print('---------------- Processing the data complete. ---------------------\n\n')
print('---------------- Saving dataset to disk... ---------------------')
test_dataset.save_to_disk('/home/or/Desktop/turkish/current/test')
print('---------------- Saving dataset to disk complete. ---------------------\n\n')

# print('---------------- Loading Data from disk... ---------------------')
# test_dataset = load_from_disk('/home/or/Desktop/turkish/current/test')
# print('---------------- Loading Data from disk complete. ---------------------\n\n')

print('\n---------------- Inferencing... ---------------------')
inputs = processor(test_dataset["speech"][:100], sampling_rate=16_000, return_tensors="pt", padding=True)  # .to(device)
with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)
print('---------------- Inferencing complete. ---------------------\n\n')


def same_length(references, predictions):
    # Zip the references and predictions, and filter by the condition
    # where the lengths are the same
    same_lengths = [(x, y) for x, y in zip(references, predictions) if len(x.split()) == len(y.split())]

    # Unzip the same_lengths list into two separate lists
    references_same_length, predictions_same_length = zip(*same_lengths)

    return list(references_same_length), list(predictions_same_length)


print('---------------- extracting same length samples... ---------------------')
references, predictions = same_length(predicted_sentences, test_dataset['sentence'][:100])
print('\n---------------- extracting same length samples complete... -------------\n\n')

results_wer = []
results_cer = []

for i, predicted_sentence in enumerate(predictions):
    print(str(i) + "-" * 100)
    print("Reference:", references[i])
    print("Prediction:", predicted_sentence)

    result_cer = cer.compute(predictions=[predicted_sentence], references=[references[i]])
    result_wer = wer.compute(predictions=[predicted_sentence], references=[references[i]])
    print("WER: " + str(result_wer))
    print("CER: " + str(result_cer))
    results_wer.append(result_wer)
    results_cer.append(result_cer)

# for i, predicted_sentence in enumerate(predicted_sentences):
#     print(str(i) + "-" * 100)
#     print("Reference:", test_dataset[i]["sentence"])
#     print("Prediction:", predicted_sentence)
#
#     result_cer = cer.compute(predictions=[predicted_sentence], references=[test_dataset[i]["sentence"]])
#     result_wer = wer.compute(predictions=[predicted_sentence], references=[test_dataset[i]["sentence"]])
#     print("WER: " + str(result_wer))
#     print("CER: " + str(result_cer))
#     results_cer.append(result_cer)
#     results_wer.append(result_wer)

num_of_bad_sample_wer = 0
num_of_sample_wer = 100
num_of_bad_sample_cer = 0
num_of_sample_cer = 100

for i in results_cer:
    if i >= 1.0:
        results_cer.remove(i)
        num_of_bad_sample_cer += 1

for i in results_wer:
    if i >= 1.0:
        results_wer.remove(i)
        num_of_bad_sample_wer += 1

num_of_sample_cer -= num_of_bad_sample_cer
num_of_sample_wer -= num_of_bad_sample_wer
print("Final WER: " + str(sum(results_wer) / num_of_sample_wer))
print("Final CER: " + str(sum(results_cer) / num_of_sample_cer))
