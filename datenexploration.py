import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import datasets

datasets.logging.set_verbosity_error()

CATEGORIES = ["All_Beauty", "Home_and_Kitchen", "Tools_and_Home_Improvement"]

def load_category(cat: str, split: str = "full", n: int | None = None) -> pd.DataFrame:
    """Lädt n Reviews einer Kategorie per Streaming und gibt ein DataFrame zurück."""

    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{cat}",
        split=split,
        streaming=True,
        trust_remote_code=True
    )

    if n is not None:
        ds = ds.take(n)

    rows = []
    for item in ds:
        rows.append({
            "rating": item["rating"],
            "text": item["text"],
            "title": item["title"],
        })

    return pd.DataFrame(rows)

dfs = {}
for cat in CATEGORIES:
    # Laden der Kategorien nacheinander und speichern im Dictionary
    dfs[cat] = load_category(cat, n=25_000)

print("Datenexploration Done")

