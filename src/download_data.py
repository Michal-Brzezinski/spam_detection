import os
import shutil
import kagglehub
import pandas as pd

# 1. Pobranie i automatyczne rozpakowanie zbioru
sciezka_cache = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
print("Pliki rozpakowane w:", sciezka_cache)

# (Opcjonalnie) Skopiowanie pliku spam.csv do obecnego folderu projektu
os.makedirs("data", exist_ok=True)
shutil.copy(os.path.join(sciezka_cache, "spam.csv"), "data/spam.csv")

# 2. Wczytanie danych (ten konkretny plik wymaga kodowania 'latin1')
df = pd.read_csv("data/spam.csv", encoding="latin1")
print(df.head())
