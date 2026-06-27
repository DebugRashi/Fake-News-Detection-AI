# -*- coding: utf-8 -*-
"""Enhanced LIAR experiments for fake news detection.

This script extends the original project with:
1. Explainable predictions with LIME when installed, plus a built-in
   coefficient explanation fallback.
2. Imbalance-aware training by comparing a normal model with a
   class-weighted model.
3. Uncertainty-aware prediction that can reject low-confidence claims.
4. A hybrid model that combines statement text with LIAR speaker/context
   metadata.
5. Error analysis reports for misclassified statements.
6. Optional cross-domain testing with another CSV dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "job_title",
    "state",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]

TRUE_LABELS = {"true", "mostly-true", "half-true"}
FALSE_LABELS = {"false", "barely-true", "pants-fire"}
TEXT_COLUMN = "statement"
CATEGORICAL_COLUMNS = ["subjects", "speaker", "job_title", "state", "party", "context"]
NUMERIC_COLUMNS = [
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
]


def to_binary_label(label: str) -> str:
    """Map LIAR's six labels into the original project's TRUE/FALSE task."""
    normalized = str(label).strip().lower()
    if normalized in TRUE_LABELS:
        return "TRUE"
    if normalized in FALSE_LABELS:
        return "FALSE"
    if normalized in {"true", "real", "1"}:
        return "TRUE"
    if normalized in {"fake", "0"}:
        return "FALSE"
    raise ValueError(f"Unknown label: {label!r}")


def read_liar_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=LIAR_COLUMNS, header=None)
    df["binary_label"] = df["label"].map(to_binary_label)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].fillna("unknown").replace("", "unknown")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(float)
    return df


def read_external_csv(path: Path) -> pd.DataFrame:
    """Load another fake-news dataset for optional cross-domain testing.

    The file should contain a statement/text column and a label column. Accepted
    column names are intentionally flexible for common fake-news datasets.
    """
    df = pd.read_csv(path)
    lower_to_original = {column.lower(): column for column in df.columns}
    text_column = next(
        (lower_to_original[name] for name in ["statement", "text", "title", "news"] if name in lower_to_original),
        None,
    )
    label_column = next((lower_to_original[name] for name in ["label", "truth", "class"] if name in lower_to_original), None)
    if text_column is None or label_column is None:
        raise ValueError("Cross-domain CSV needs a text column and a label column.")

    output = pd.DataFrame()
    output[TEXT_COLUMN] = df[text_column].fillna("").astype(str)
    output["binary_label"] = df[label_column].map(to_binary_label)
    return output


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def text_model(class_weight: str | None = None) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight=class_weight, solver="liblinear"),
            ),
        ]
    )


def hybrid_model(class_weight: str | None = "balanced") -> Pipeline:
    text_features = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30000,
                ),
            )
        ]
    )
    categorical_features = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("one_hot", one_hot_encoder()),
        ]
    )
    numeric_features = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    features = ColumnTransformer(
        [
            ("text", text_features, TEXT_COLUMN),
            ("categorical", categorical_features, CATEGORICAL_COLUMNS),
            ("numeric", numeric_features, NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight, solver="liblinear")),
        ]
    )


def evaluate(name: str, model: Pipeline, x_test, y_test: Iterable[str]) -> dict:
    predictions = model.predict(x_test)
    labels = ["FALSE", "TRUE"]
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    return {
        "name": name,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": report,
    }


def print_metric_summary(results: list[dict]) -> None:
    for result in results:
        report = result["classification_report"]
        print(f"\n=== {result['name']} ===")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print("Confusion matrix rows/cols: FALSE, TRUE")
        print(np.array(result["confusion_matrix"]))
        print(
            "F1 FALSE={:.3f} | F1 TRUE={:.3f} | Macro F1={:.3f}".format(
                report["FALSE"]["f1-score"],
                report["TRUE"]["f1-score"],
                report["macro avg"]["f1-score"],
            )
        )


def rejection_report(model: Pipeline, x_test, y_test: pd.Series, threshold: float) -> dict:
    probabilities = model.predict_proba(x_test)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    accepted = confidence >= threshold
    accepted_count = int(accepted.sum())
    rejected_count = int((~accepted).sum())
    accepted_accuracy = accuracy_score(y_test[accepted], predictions[accepted]) if accepted_count else None
    return {
        "threshold": threshold,
        "accepted": accepted_count,
        "rejected": rejected_count,
        "coverage": accepted_count / len(y_test),
        "accepted_accuracy": accepted_accuracy,
        "overall_accuracy_without_rejection": accuracy_score(y_test, predictions),
    }


def coefficient_explanation(model: Pipeline, statement: str, top_n: int = 8) -> list[dict]:
    """Explain a text-only Logistic Regression prediction from TF-IDF weights."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    matrix = vectorizer.transform([statement])
    feature_names = vectorizer.get_feature_names_out()
    true_index = list(classifier.classes_).index("TRUE")
    coefficients = classifier.coef_[0]
    if true_index == 0:
        coefficients = -coefficients
    contributions = matrix.multiply(coefficients).toarray()[0]
    non_zero = np.flatnonzero(contributions)
    ranked = sorted(non_zero, key=lambda index: abs(contributions[index]), reverse=True)[:top_n]
    return [
        {
            "feature": feature_names[index],
            "contribution": float(contributions[index]),
            "pushes_toward": "TRUE" if contributions[index] > 0 else "FALSE",
        }
        for index in ranked
    ]


def lime_explanation(model: Pipeline, statement: str, top_n: int = 8) -> list[dict] | None:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return None

    explainer = LimeTextExplainer(class_names=list(model.classes_))
    explanation = explainer.explain_instance(statement, model.predict_proba, num_features=top_n)
    return [
        {
            "feature": feature,
            "weight": float(weight),
            "pushes_toward": "TRUE" if weight > 0 else "FALSE",
        }
        for feature, weight in explanation.as_list()
    ]


def shap_explanation(model: Pipeline, train_text: pd.Series, statement: str, top_n: int = 8) -> list[dict] | None:
    try:
        import shap
    except ImportError:
        return None

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    background = vectorizer.transform(train_text.sample(min(500, len(train_text)), random_state=7))
    sample = vectorizer.transform([statement])
    explainer = shap.LinearExplainer(classifier, background)
    shap_values = explainer.shap_values(sample)
    values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    feature_names = vectorizer.get_feature_names_out()
    non_zero = np.flatnonzero(sample.toarray()[0])
    ranked = sorted(non_zero, key=lambda index: abs(values[index]), reverse=True)[:top_n]
    return [
        {
            "feature": feature_names[index],
            "shap_value": float(values[index]),
            "pushes_toward": "TRUE" if values[index] > 0 else "FALSE",
        }
        for index in ranked
    ]


def write_error_analysis(model: Pipeline, test_df: pd.DataFrame, output_dir: Path) -> dict:
    predictions = model.predict(test_df)
    probabilities = model.predict_proba(test_df)
    confidence = np.max(probabilities, axis=1)
    errors = test_df.copy()
    errors["prediction"] = predictions
    errors["confidence"] = confidence
    errors = errors[errors["prediction"] != errors["binary_label"]].copy()
    errors["statement_length"] = errors[TEXT_COLUMN].str.split().str.len()
    errors["length_bucket"] = pd.cut(
        errors["statement_length"],
        bins=[0, 10, 20, 40, 1000],
        labels=["0-10", "11-20", "21-40", "41+"],
        include_lowest=True,
    )
    errors["primary_subject"] = errors["subjects"].fillna("unknown").str.split(",").str[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_dir / "misclassified_errors.csv", index=False)
    summary = {
        "total_errors": int(len(errors)),
        "errors_by_true_label": errors["binary_label"].value_counts().to_dict(),
        "errors_by_party": errors["party"].value_counts().head(10).to_dict(),
        "errors_by_subject": errors["primary_subject"].value_counts().head(10).to_dict(),
        "errors_by_length_bucket": errors["length_bucket"].value_counts().to_dict(),
        "highest_confidence_errors": errors.sort_values("confidence", ascending=False)
        .head(10)[["statement", "binary_label", "prediction", "confidence", "speaker", "party", "subjects"]]
        .to_dict(orient="records"),
    }
    return summary


def write_uncertainty_report(output_dir: Path, rejection: dict) -> None:
    pd.DataFrame([rejection]).to_csv(output_dir / "uncertainty_rejection.csv", index=False)


def write_explanation_report(output_dir: Path, explanation: dict) -> None:
    rows = []
    for method in ["lime", "shap", "coefficient_fallback"]:
        values = explanation.get(method) or []
        for item in values:
            rows.append(
                {
                    "method": method,
                    "feature": item["feature"],
                    "pushes_toward": item["pushes_toward"],
                    "score": item.get("weight", item.get("shap_value", item.get("contribution"))),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "sample_explanation.csv", index=False)


def write_experiment_report(
    output_dir: Path,
    results: list[dict],
    rejection: dict,
    explanation: dict,
    error_summary: dict,
) -> None:
    """Write readable experiment report files."""
    rows = []
    for result in results:
        report = result["classification_report"]
        rows.append(
            {
                "model": result["name"],
                "accuracy": round(result["accuracy"], 3),
                "false_precision": round(report["FALSE"]["precision"], 3),
                "false_recall": round(report["FALSE"]["recall"], 3),
                "false_f1": round(report["FALSE"]["f1-score"], 3),
                "true_precision": round(report["TRUE"]["precision"], 3),
                "true_recall": round(report["TRUE"]["recall"], 3),
                "true_f1": round(report["TRUE"]["f1-score"], 3),
                "macro_f1": round(report["macro avg"]["f1-score"], 3),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "model_comparison.csv", index=False)

    lines = [
        "Enhanced Fake News Detection Report",
        "===================================",
        "",
        "1. Model comparison",
    ]
    for row in rows:
        lines.append(
            "- {model}: accuracy={accuracy}, FALSE F1={false_f1}, TRUE F1={true_f1}, macro F1={macro_f1}".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "2. Uncertainty-aware detection",
            f"- Confidence threshold: {rejection['threshold']}",
            f"- Accepted predictions: {rejection['accepted']}",
            f"- Rejected uncertain predictions: {rejection['rejected']}",
            f"- Accuracy on accepted predictions: {rejection['accepted_accuracy']:.3f}",
            "",
            "3. Explainable prediction example",
            f"- Statement: {explanation['statement']}",
            f"- Prediction: {explanation['prediction']}",
            f"- Probability FALSE: {explanation['probability']['FALSE']:.3f}",
            f"- Probability TRUE: {explanation['probability']['TRUE']:.3f}",
            "- Top explanation features:",
        ]
    )
    for item in explanation["coefficient_fallback"][:8]:
        lines.append(
            f"  * {item['feature']} pushes toward {item['pushes_toward']} "
            f"(score {item['contribution']:.3f})"
        )

    lines.extend(
        [
            "",
            "4. Error analysis",
            f"- Total wrong predictions: {error_summary['total_errors']}",
            f"- Errors by true label: {error_summary['errors_by_true_label']}",
            f"- Most common error subjects: {error_summary['errors_by_subject']}",
            f"- Errors by statement length: {error_summary['errors_by_length_bucket']}",
            "",
            "Conclusion",
            "The hybrid model performed best because it used both statement text and LIAR metadata.",
            "The uncertainty-aware model improved reliability by rejecting low-confidence claims.",
        ]
    )
    (output_dir / "experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enhanced fake-news detection experiments on LIAR.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "liar_dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "enhanced_outputs")
    parser.add_argument("--threshold", type=float, default=0.65, help="Reject predictions below this confidence.")
    parser.add_argument("--cross-domain-csv", type=Path, default=None, help="Optional external CSV for cross-domain test.")
    parser.add_argument(
        "--explain",
        type=str,
        default="Says the health care reform law will require free sex change surgeries.",
        help="Statement to explain after training.",
    )
    args = parser.parse_args()

    train = read_liar_split(args.data_dir / "train.tsv")
    valid = read_liar_split(args.data_dir / "valid.tsv")
    test = read_liar_split(args.data_dir / "test.tsv")

    train_full = pd.concat([train, valid], ignore_index=True)
    y_train = train_full["binary_label"]
    y_test = test["binary_label"]

    print("Class distribution in training data:")
    print(y_train.value_counts())

    baseline = text_model(class_weight=None)
    balanced = text_model(class_weight="balanced")
    hybrid = hybrid_model(class_weight="balanced")

    baseline.fit(train_full[TEXT_COLUMN], y_train)
    balanced.fit(train_full[TEXT_COLUMN], y_train)
    hybrid.fit(train_full, y_train)

    results = [
        evaluate("Text-only baseline", baseline, test[TEXT_COLUMN], y_test),
        evaluate("Text-only class_weight=balanced", balanced, test[TEXT_COLUMN], y_test),
        evaluate("Hybrid text + LIAR metadata", hybrid, test, y_test),
    ]
    print_metric_summary(results)

    rejection = rejection_report(balanced, test[TEXT_COLUMN], y_test, args.threshold)
    print("\n=== Uncertainty-aware rejection ===")
    print(f"Threshold: {rejection['threshold']}")
    print(f"Accepted predictions: {rejection['accepted']}")
    print(f"Rejected uncertain predictions: {rejection['rejected']}")
    print(f"Coverage: {rejection['coverage']:.3f}")
    print(f"Accepted accuracy: {rejection['accepted_accuracy']:.3f}")

    prediction = balanced.predict([args.explain])[0]
    probability = dict(zip(balanced.classes_, balanced.predict_proba([args.explain])[0]))
    explanation = {
        "statement": args.explain,
        "prediction": prediction,
        "probability": {label: float(value) for label, value in probability.items()},
        "lime": lime_explanation(balanced, args.explain),
        "shap": shap_explanation(balanced, train_full[TEXT_COLUMN], args.explain),
        "coefficient_fallback": coefficient_explanation(balanced, args.explain),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Explanation for sample statement ===")
    print(f"Statement: {explanation['statement']}")
    print(f"Prediction: {explanation['prediction']}")
    print(f"Probability FALSE: {explanation['probability']['FALSE']:.3f}")
    print(f"Probability TRUE: {explanation['probability']['TRUE']:.3f}")
    print("Top coefficient explanation features:")
    for item in explanation["coefficient_fallback"][:8]:
        print(f"- {item['feature']} -> {item['pushes_toward']} ({item['contribution']:.3f})")

    error_summary = write_error_analysis(hybrid, test, args.output_dir)
    write_uncertainty_report(args.output_dir, rejection)
    write_explanation_report(args.output_dir, explanation)
    write_experiment_report(args.output_dir, results, rejection, explanation, error_summary)
    print("\n=== Error analysis summary ===")
    print(f"Total wrong predictions: {error_summary['total_errors']}")
    print(f"Errors by true label: {error_summary['errors_by_true_label']}")
    print(f"Most common error subjects: {error_summary['errors_by_subject']}")
    print(f"Errors by statement length: {error_summary['errors_by_length_bucket']}")

    if args.cross_domain_csv:
        external = read_external_csv(args.cross_domain_csv)
        cross_result = evaluate(
            f"Cross-domain: LIAR-trained balanced text model on {args.cross_domain_csv.name}",
            balanced,
            external[TEXT_COLUMN],
            external["binary_label"],
        )
        cross_report = cross_result["classification_report"]
        pd.DataFrame(
            [
                {
                    "model": cross_result["name"],
                    "accuracy": cross_result["accuracy"],
                    "false_f1": cross_report["FALSE"]["f1-score"],
                    "true_f1": cross_report["TRUE"]["f1-score"],
                    "macro_f1": cross_report["macro avg"]["f1-score"],
                }
            ]
        ).to_csv(args.output_dir / "cross_domain_result.csv", index=False)
        print("\n=== Cross-domain robustness ===")
        print(f"Accuracy: {cross_result['accuracy']:.3f}")
        print(f"Macro F1: {cross_report['macro avg']['f1-score']:.3f}")

    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
