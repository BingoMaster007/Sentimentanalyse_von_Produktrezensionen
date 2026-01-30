from datenexploration import dfs
df_raw = dfs["All_Beauty"]

import pandas as pd
import re
df_raw = dfs["All_Beauty"]

"""Überführen von Titel und Text in einen Text. 
fehlende Werte werden durch Leere Strings ersetzt.
Überflüssige Leerzeichen werden entfernt"""
def build_full_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["full_text"] = (df["title"].str.strip() + " " + df["text"].str.strip()).str.strip()
    return df

df_clean = build_full_text(df_raw)
df_clean["full_text"].head(5)
_non_alpha = re.compile(r"[^a-zA-Z\s]+")

def clean_text_series(s: pd.Series) -> pd.Series:
    """Alles klein schreiben, Sonderzeichen durch Leerzeichen ersetzen und
    mehrere Leerzeichen zu einem machen"""
    s = s.str.lower()
    s = s.str.replace(_non_alpha, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

pattern = re.compile(r"[^a-z\s]")
df_clean["full_text"].str.contains(pattern).any()

#%%
def add_word_count(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    """Spaltet den Text in Wörter und zählt sie"""
    df = df.copy()
    df["word_count"] = df[text_col].fillna("").str.split().str.len()
    return df

def filter_short_reviews(df: pd.DataFrame, min_words: int = 5, text_col: str = "full_text") -> pd.DataFrame:
    """Entfernen von Reviews mit wenigre als 5 Wörtern"""
    df = add_word_count(df, text_col=text_col)
    return df[df["word_count"] >= min_words].copy()

#%%
def preprocess_df(df: pd.DataFrame, min_words: int = 5) -> pd.DataFrame:
    df = build_full_text(df)
    df["full_text"] = clean_text_series(df["full_text"])
    df["word_count"] = df["full_text"].str.split().str.len()
    df = df[df["word_count"] >= min_words].copy()
    df["rating"] = df["rating"].astype(int)
    return df


dfs_clean = {}
for cat, df in dfs.items():
    """Auf alle Kategorien anwenden und Kontrolle"""
    before = len(df)
    df_clean = preprocess_df(df, min_words=5)
    after = len(df_clean)
    print(f"{cat}: vorher={before}, nachher={after}, entfernt={before-after} ({(before-after)/before*100:.2f}%)")
    dfs_clean[cat] = df_clean

cat = "All_Beauty"

df_raw = dfs[cat]
df_clean = dfs_clean[cat]

df_clean.dtypes

df_all = pd.concat(dfs_clean.values(), ignore_index=True)

print(len(df_all))

X = df_all["full_text"]
y = df_all["rating"]

from sklearn.model_selection import train_test_split
"""X=Features, Y=Labels, test_size=0.2-> 20% der Daten sind Testdaten,
stratify=y -> löst das Problem des Klassenungleichgewichts,
random_state=42 -> Garantiert selben Startwert bei jedem Lauf"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
"""Bigrams ermöglichen training von Wortpaaren mit z.B. Negationen; max. features begrenzt das Vokabular um Rechenleistung zu minimieren; ein Wortpaar wird nur dann als feature aufgenommen wenn es in mindestens 5 Bewertungen vorkommt; englische Stoppwörter werden entfernt"""
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10_000,
    min_df=5,
    stop_words="english"
)
#%%
X_train_tfidf = tfidf.fit_transform(X_train)  # lernt Vokabular + erstellt Matrix
X_test_tfidf  = tfidf.transform(X_test)       # nutzt gleiches Vokabular
#%%
"""Zeigt Anzahl der Reviews die für das Training genutzt wurden und die Anzahl der Features."""
print(X_train_tfidf.shape)
