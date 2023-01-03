import warnings

import evaluate as evaluate
import torch
import re
import jiwer
import torchaudio
from datasets import load_dataset, load_metric, Features, Value, Audio, load_from_disk
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

'''
One important note -> the wer that you see during training as the validation wer is often significantly 
worse than the actual “Test WER” because during training the data is processed in a way that is ideal 
for the model to learn better (appending the " " at the end of every sample), 
but not the correct preprocessing when running the actual eval. 
So it’s important to always run the eval script (that can be copy-pasted from the template) once 
and use this final WER as the result.
'''

MODEL_ID = "../portu_clean/checkpoint_4488"
DEVICE = "cuda"
# DATASET = "mozilla-foundation/common_voice_11_0"

print("----------------- Checking if cuda is available... -----------------")
print(f'Cuda Available = {torch.cuda.is_available()}\n\n')

# features = Features(
#     {
#         "client_id": Value("string"),
#         "path": Value("string"),
#         "audio": Audio(sampling_rate=48_000),
#         "sentence": Value("string"),
#         "up_votes": Value("int64"),
#         "down_votes": Value("int64"),
#         "age": Value("string"),
#         "gender": Value("string"),
#         "accents": Value("string"),
#         "locale": Value("string"),
#         "segment": Value("string"),
#     }
# )
#
# print("----------------- Loading Datasets... -----------------")
# sample_data = load_dataset('csv', data_files={'test': 'test.csv', }, data_dir='/home/or/Desktop/arabic_new_dataset')
# print("----------------- Loading Datasets complete. ----------\n\n")
#
# print("----------------- Casting features... -----------------")
# sample_data = sample_data.cast(features)
# print("----------------- Casting features complete. -----------\n\n")
#
# print("----------------- Removing columns... -----------------")
# sample_data = sample_data.remove_columns(["accents", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
# print("----------------- Removing columns complete. -----------\n\n")
#
# print("----------------- Loading audio from path... -----------------")
# test_dataset = sample_data['test']
# test_dataset['audio']
# print("----------------- Loading audio from path complete. -----------\n\n")

CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ"]


print("----------------- Loading Metrics... -----------------")
wer = evaluate.load("wer")
cer = evaluate.load("cer")
print("----------------- Loading Metrics complete. -----------------\n\n")

chars_to_ignore_regex = f"[{re.escape(''.join(CHARS_TO_IGNORE))}]"
print('---------------- Loading Data from disk... ---------------------')
test_dataset = load_from_disk('/home/or/Desktop/portu/current/test')
print('---------------- Loading Data from disk complete. ---------------------\n\n')
tokenizer = Wav2Vec2CTCTokenizer("../vocab_portu.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained('../portu_clean/checkpoint-4488')
model.to(DEVICE)

# Preprocessing the datasets.
# We need to read the audio files as arrays
resampler = torchaudio.transforms.Resample(48_000, 16_000)


def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    batch["sentence"] = re.sub(chars_to_ignore_regex, "", batch["sentence"]).lower()
    return batch


print("----------------- Preparing datasets... -----------------")
test_dataset = test_dataset.map(speech_file_to_array_fn)
print("----------------- Preparing datasets complete. -----------------\n\n")


# Preprocessing the datasets.
# We need to read the audio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch


print("----------------- Processing the data... -----------------")
result = test_dataset.map(evaluate, batched=False)
print("----------------- Processing the data complete. -----------------\n\n")

predictions = [x.lower() for x in result["pred_strings"]]
references = [x.lower() for x in result["sentence"]]

print(f"WER: {wer.compute(predictions=predictions, references=references) * 100}")
print(f"CER: {cer.compute(predictions=predictions, references=references,) * 100}")

# results_wer = []
# results_cer = []
#
# print("----------------- Computing... -----------------")
# for i, predicted_sentence in enumerate(predictions):
#     results_wer = wer.compute(predictions=predicted_sentence, references=result[i]['sentence'])
#     results_cer = cer.compute(predictions=predicted_sentence, references=result[i]['sentence'])
#     print("Result: " + str(result))
#     results_wer.append(results_wer)
#     results_cer.append(results_cer)
# print("----------------- Computing complete. -----------------\n\n")
# num_of_bad_sample = 0
#
#
# for i in results_cer:
#     if i >= 1.0:
#         results_wer.remove(i)
#         results_cer.remove(i)
#         num_of_bad_sample += 1
#
#
# print("CER: " + str(sum(results_cer) / len(results_cer)))
# print("WER: " + str(sum(results_cer) / len(results_cer)))