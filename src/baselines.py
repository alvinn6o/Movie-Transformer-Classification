"""Non-transformer reference points.

    python -m src.baselines
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_dataset
from src.evaluate import binary_metrics, pick_threshold, save_run


def main():
    df = load_dataset()
    split = {s: df[df["split"] == s] for s in ("train", "dev", "test")}
    y = {s: split[s]["label"].to_numpy() for s in split}

    # Majority class: constant score, so ranking metrics sit at chance.
    prior = y["train"].mean()
    save_run("binary", "majority", 0, {"inputs": "none", "trainable_params": 0},
             binary_metrics(y["dev"], np.full(len(y["dev"]), prior), 0.5),
             binary_metrics(y["test"], np.full(len(y["test"]), prior), 0.5))

    # TF-IDF on plot text; vocabulary fit on train only, C chosen on dev.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = {"train": vec.fit_transform(split["train"]["plot"])}
    X.update({s: vec.transform(split[s]["plot"]) for s in ("dev", "test")})
    search = {}
    for C in [0.25, 1.0, 4.0, 16.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X["train"], y["train"])
        search[str(C)] = binary_metrics(y["dev"], clf.predict_proba(X["dev"])[:, 1], 0.5)["average_precision"]
    C = float(max(search, key=search.get))
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X["train"], y["train"])
    dev_s, test_s = (clf.predict_proba(X[s])[:, 1] for s in ("dev", "test"))
    threshold = pick_threshold(y["dev"], dev_s)
    test = binary_metrics(y["test"], test_s, threshold)
    save_run("binary", "tfidf_logreg", 0,
             {"inputs": "plot", "C": C, "C_search": search, "vocab_size": len(vec.vocabulary_),
              "trainable_params": int(clf.coef_.size + 1)},
             binary_metrics(y["dev"], dev_s, threshold), test,
             {"dev_scores": dev_s, "dev_labels": y["dev"], "test_scores": test_s, "test_labels": y["test"]})
    print(f"tfidf_logreg  C={C}  test AP {test['average_precision']:.3f}  F1 {test['f1']:.3f}")


if __name__ == "__main__":
    main()
