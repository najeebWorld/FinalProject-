from datasets import load_from_disk

print("----------------- Loading Datasets... -----------------")
data = load_from_disk('/home/or/Desktop/arabic_new_dataset/arrow/test')
print("----------------- Loading Datasets complete. -----------------\n\n")

print(data.column_names)
print(data[0])



