from pathlib import Path
import pandas as pd

# Buduje ścieżkę: katalog_skryptu (src) -> poziom wyżej (P4) -> data -> spam.csv
csv_path = Path(__file__).resolve().parent.parent / "data" / "spam.csv"

df = pd.read_csv(
    csv_path,
    encoding="latin1",
    usecols=[0, 1],          # bierze tylko dwie pierwsze kolumny
    header=0,                # nadpisuje pierwszy wiersz (v1, v2)
    names=["label", "text"]
)

print(df.head())
print(df.shape)
print(df["label"].value_counts())