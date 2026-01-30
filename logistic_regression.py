from datenvorverarbeitung import X_train_tfidf, X_test_tfidf, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

"""max. Anzahl an Iterationen festgelegt; seltene KLassen  bekommen mehr Gewicht; 
solver=algorithmus der die beste Gewichtung findet; nutzt die gesamte CPU Kapazität"""
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
)
log_reg.fit(X_train_tfidf, y_train)
y_pred = log_reg.predict(X_test_tfidf)
print(log_reg.classes_)

"""Evaluationsmetriken:"""

"""Wie viel % der Bewertungen wurden korrekt vorhergesagt"""
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

"""Liefert folgende KPIs pro Klasse: Precision, Recall, F1-Score, Macro-Average, 
Micro-Average, Weighted-Average"""
report = classification_report(y_test, y_pred)
print(report)

# Konfusionsmatrix berechnen
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=log_reg.classes_,
    yticklabels=log_reg.classes_
)
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.title("Konfusionsmatrix: 75.000 Bewertungen gesamt")
plt.show()

