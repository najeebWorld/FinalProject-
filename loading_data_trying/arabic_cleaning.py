import pyarabic.araby as araby

before_filter = "وَرَوَى حُمَيْدُ عَنْ أَنَاسٍ أَنَّ النَّبِيَّ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ قَالَ"
after_filter = araby.strip_diacritics(before_filter)

print("Before clean: " + before_filter)
print("After clean: " + after_filter)
