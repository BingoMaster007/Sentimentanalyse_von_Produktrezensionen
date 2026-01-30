from datenvorverarbeitung import df_all
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def map_rating_to_sentiment(rating: int) -> int:
    if rating <= 2:
        return 0      # negativ
    elif rating == 3:
        return 1      # neutral
    else:
        return 2      # positiv

df_all["sentiment"] = df_all["rating"].apply(map_rating_to_sentiment)

X = df_all["full_text"]
y = df_all["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
label_names = ["negativ", "neutral", "positiv"]
label_ids = [0, 1, 2]

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10_000,
    min_df=5,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)  # lernt Vokabular + erstellt Matrix
X_test_tfidf  = tfidf.transform(X_test)       # nutzt gleiches Vokabular

log_reg.fit(X_train_tfidf, y_train)
y_pred = log_reg.predict(X_test_tfidf)

print("Klassen (IDs):", log_reg.classes_)  # sollte [0 1 2] sein

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(
    y_test, y_pred,
    labels=label_ids,
    target_names=label_names
)
print(report)

cm = confusion_matrix(y_test, y_pred, labels=label_ids)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.title("Konfusionsmatrix: 3 Klassen (negativ/neutral/positiv)")
plt.show()
