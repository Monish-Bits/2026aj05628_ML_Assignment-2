"""
Streamlit front-end for ML Assignment 2 - Student Dropout & Academic Success.

The app never retrains anything: it loads the five pipelines saved by
train_models.py from model/, scores the uploaded test CSV and reports the
six evaluation metrics, the confusion matrix and the classification report.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

PRIMARY = "#3B5BDB"
MUTED = "#ADB5BD"
# one colour per outcome, used in every chart so a class always looks the same
CLASS_COLORS = {"Dropout": "#E03131", "Enrolled": "#F08C00", "Graduate": "#2F9E44"}

MODEL_BLURBS = {
    "logistic_regression": "Linear model on one-hot + standardised features, class-balanced.",
    "decision_tree": "Single CART tree; leaf-size limit controls overfitting.",
    "knn": "Votes of the nearest students in the scaled feature space.",
    "naive_bayes": "Gaussian Naive Bayes; assumes features are independent per class.",
    "svm": "RBF-kernel SVM, class-balanced, with Platt-calibrated probabilities.",
}

METRIC_HELP = {
    "Accuracy": "Share of all students whose outcome was predicted correctly.",
    "AUC": "One-vs-rest ROC-AUC, macro-averaged: how well the predicted probabilities "
           "rank each class above the others (0.5 = random, 1 = perfect).",
    "Precision": "Macro average over classes: of the students predicted as a class, "
                 "how many really belong to it.",
    "Recall": "Macro average over classes: of the students in a class, how many the "
              "model found.",
    "F1": "Macro-averaged harmonic mean of precision and recall - the tuning objective.",
    "MCC": "Matthews correlation between actual and predicted labels (-1 to 1); "
           "robust to the class imbalance.",
}

st.set_page_config(page_title="Student Outcome Classifier", page_icon="🎓", layout="wide")

st.html("""
<style>
.hero {
  padding: 1.6rem 1.8rem; border-radius: 0.9rem; margin-bottom: 0.4rem;
  background: linear-gradient(120deg, rgba(59,91,219,0.14), rgba(47,158,68,0.10));
  border: 1px solid rgba(59,91,219,0.25);
}
.hero h1 { font-size: 1.85rem; margin: 0 0 0.35rem 0; padding: 0; }
.hero p { margin: 0; opacity: 0.85; font-size: 1.02rem; max-width: 60rem; }
.chips { margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
.chip {
  font-size: 0.82rem; padding: 0.18rem 0.7rem; border-radius: 999px;
  background: rgba(127,127,127,0.12); border: 1px solid rgba(127,127,127,0.25);
}
/* mark the active entry in open dropdown lists */
[role="option"][aria-selected="true"] { font-weight: 600; color: #3B5BDB; }
[role="option"][aria-selected="true"]::after { content: "✓"; margin-left: auto; padding-left: 0.75rem; }
.step { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #3B5BDB; margin-bottom: 0.1rem; }
</style>
""")


# ---------------------------------------------------------------- helpers
@st.cache_data
def load_metadata() -> dict:
    return json.loads((MODEL_DIR / "metadata.json").read_text())


@st.cache_resource
def load_models(model_keys: tuple) -> dict:
    return {key: joblib.load(MODEL_DIR / f"{key}.joblib") for key in model_keys}


def compute_metrics(model, X, y_true) -> dict:
    """Same definitions as train_models.evaluate(): macro averages, one-vs-rest AUC."""
    y_pred = model.predict(X)
    try:
        auc = roc_auc_score(y_true, model.predict_proba(X), multi_class="ovr",
                            average="macro", labels=model.classes_)
    except ValueError:
        # AUC is undefined when the uploaded file holds a single class only
        auc = np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner="Scoring the test data with all five models…")
def score_all_models(X: pd.DataFrame, y: pd.Series, _models: dict) -> dict:
    """Predictions + metrics for every saved model; cached so switching models is instant."""
    out = {}
    for key, model in _models.items():
        out[key] = {
            "metrics": compute_metrics(model, X, y),
            "pred": model.predict(X),
            "proba": pd.DataFrame(model.predict_proba(X), columns=model.classes_, index=X.index),
        }
    return out


def chart_style():
    """Transparent figures with mid-grey ink, readable on both light and dark themes."""
    ink, grid = "#868E96", "#868E9655"
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": ink, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
        "axes.edgecolor": grid, "grid.color": grid, "font.size": 10,
    })


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    annot = np.array([[f"{n}\n{p:.0f}%" for n, p in zip(r_n, r_p)]
                      for r_n, r_p in zip(cm, row_pct)])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(row_pct, annot=annot, fmt="", cmap="Blues",
                vmin=0, vmax=100, xticklabels=labels, yticklabels=labels, cbar=False,
                linewidths=2, linecolor=(0, 0, 0, 0), annot_kws={"fontsize": 11},
                square=True, ax=ax)
    ax.set_xlabel("Predicted outcome", labelpad=8)
    ax.set_ylabel("Actual outcome", labelpad=8)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig


def plot_ranking(table: pd.DataFrame, metric: str, highlight: str):
    ranked = table[metric].sort_values()
    colors = [PRIMARY if name == highlight else MUTED for name in ranked.index]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    bars = ax.barh(ranked.index, ranked.values, color=colors, height=0.62)
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=9)
    lo = max(0.0, np.nanmin(ranked.values) - 0.1)
    ax.set_xlim(lo, min(1.0, np.nanmax(ranked.values) + 0.04))
    ax.set_xticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def plot_class_counts(y: pd.Series, classes):
    counts = y.value_counts().reindex(classes, fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 2.2))
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=[CLASS_COLORS.get(c, MUTED) for c in counts.index[::-1]], height=0.6)
    ax.bar_label(bars, labels=[f"{v}  ({v / counts.sum():.0%})" for v in counts.values[::-1]],
                 padding=4, fontsize=9)
    ax.set_xlim(0, counts.max() * 1.3)
    ax.set_xticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------- static data
meta = load_metadata()
MODEL_LABELS = meta["models"]                  # key -> display name
FEATURES = meta["feature_columns"]
TARGET = meta["target"]
CLASSES = meta["classes"]
models = load_models(tuple(MODEL_LABELS))
chart_style()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎓 About this app")
    st.markdown(
        "Five classifiers were trained offline on 3,539 students. "
        "This app only **loads** them and **evaluates** them on the data you "
        "provide. Nothing is retrained here."
    )
    st.markdown("**How to use**")
    st.markdown(
        "1. Upload a labelled test CSV (a sample is downloadable)\n"
        "2. Pick a model from the dropdown\n"
        "3. Explore metrics, the confusion matrix and the model comparison"
    )
    st.divider()
    st.markdown("**Metric conventions**")
    st.caption("Precision, Recall and F1 are macro-averaged over the three classes. "
               "AUC is one-vs-rest macro ROC-AUC.")
    st.markdown("[Dataset on UCI ↗](https://archive.ics.uci.edu/dataset/697)")

# ---------------------------------------------------------------- hero
st.html(f"""
<div class="hero">
  <h1>Student Dropout &amp; Academic Success</h1>
  <p>Will a student <b>drop out</b>, stay <b>enrolled</b> or <b>graduate</b>? Compare five
  pre-trained classifiers on labelled test data, using enrolment, socio-economic and
  first-year academic records.</p>
  <div class="chips">
    <span class="chip">UCI dataset #697</span>
    <span class="chip">4,424 students</span>
    <span class="chip">{len(FEATURES)} features</span>
    <span class="chip">3 outcome classes</span>
    <span class="chip">5 models · no retraining</span>
  </div>
</div>
""")

# ---------------------------------------------------------------- step 1 + 2
col_data, col_model = st.columns([1.15, 1], gap="medium")

with col_data, st.container(border=True, height="stretch"):
    st.html('<div class="step">Step 1 · Upload test data</div>')
    uploaded = st.file_uploader(
        f"CSV with all {len(FEATURES)} feature columns and a `{TARGET}` column",
        type=["csv"],
    )

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Sample test_data.csv", data=SAMPLE_CSV.read_bytes(), file_name="test_data.csv",
        mime="text/csv", icon=":material/download:", width="stretch",
        help=f"The {meta['split']['test_rows']} students held out from training "
             "(stratified 20 % split), never seen by any model.",
    )
    dl2.download_button(
        "Empty template", data=",".join(f'"{c}"' for c in FEATURES + [TARGET]) + "\n",
        file_name="template.csv", mime="text/csv", icon=":material/table_view:",
        width="stretch", help="Header row only - the exact columns the app expects.",
    )

with col_model, st.container(border=True, height="stretch"):
    st.html('<div class="step">Step 2 · Model</div>')
    # plain dropdown: display names as options, typing/filtering disabled so the box
    # always shows the current choice
    label_to_key = {label: key for key, label in MODEL_LABELS.items()}
    chosen_label = st.selectbox(
        "Classification model",
        options=list(label_to_key),
        index=list(label_to_key).index(MODEL_LABELS["svm"]),
        key="model_choice",
        filter_mode=None,
        label_visibility="collapsed",
    )
    chosen_key = label_to_key[chosen_label]
    st.caption(MODEL_BLURBS[chosen_key])
    params = meta["best_params"][chosen_key]
    st.markdown("Tuned hyper-parameters (5-fold CV):  \n" +
                "  ".join(f"`{k} = {v}`" for k, v in params.items()))

# ---------------------------------------------------------------- load + validate
if uploaded is None:
    st.info("Upload a labelled CSV to evaluate the models. Don't have one? Download the "
            "**sample test_data.csv** above and upload it.", icon=":material/upload_file:")
    st.stop()
try:
    data = pd.read_csv(uploaded)
except Exception as exc:  # malformed / non-CSV file
    st.error(f"Could not read `{uploaded.name}` as CSV: {exc}", icon=":material/error:")
    st.stop()
source_name = uploaded.name

data.columns = [c.strip() for c in data.columns]
missing = [c for c in FEATURES if c not in data.columns]
if TARGET not in data.columns:
    missing.append(TARGET)
if missing:
    st.error(f"`{source_name}` is missing **{len(missing)}** required column(s). "
             "Use the empty template above for the exact header.", icon=":material/error:")
    st.code("\n".join(missing), language=None)
    st.stop()

n_raw = len(data)
data = data.dropna(subset=FEATURES + [TARGET])
data[TARGET] = data[TARGET].astype(str).str.strip()
unknown = sorted(set(data[TARGET]) - set(CLASSES))
if unknown:
    data = data[data[TARGET].isin(CLASSES)]
if data.empty:
    st.error("No usable rows left after validation.", icon=":material/error:")
    st.stop()

X_eval, y_eval = data[FEATURES], data[TARGET]
results = score_all_models(X_eval, y_eval, models)
current = results[chosen_key]
scores = current["metrics"]
table = pd.DataFrame({MODEL_LABELS[k]: r["metrics"] for k, r in results.items()}).T
averages = table.mean()

# ---------------------------------------------------------------- results header
st.space("small")
head_l, head_r = st.columns([3, 2], vertical_alignment="bottom")
head_l.subheader(f"Results · {MODEL_LABELS[chosen_key]}")
head_l.caption(f"Evaluated on **{source_name}** with the saved model (no retraining).")
with head_r:
    badges = st.container(horizontal=True, horizontal_alignment="right")
    badges.badge(f"{len(data)} rows evaluated", icon=":material/check:", color="green")
    badges.badge(f"{len(FEATURES)}/{len(FEATURES)} features", color="blue")
    if n_raw - len(data):
        badges.badge(f"{n_raw - len(data)} rows skipped", icon=":material/warning:",
                     color="orange",
                     help="Rows with missing values or unknown target labels"
                          + (f" {unknown}" if unknown else "") + ".")

kpis = st.columns(6)
for col, (name, value) in zip(kpis, scores.items()):
    if np.isnan(value):
        col.metric(name, "n/a", help=METRIC_HELP[name] + " (Needs ≥ 2 classes.)", border=True)
        continue
    diff = value - averages[name]
    col.metric(name, f"{value:.4f}", delta=f"{diff:+.4f} vs avg", help=METRIC_HELP[name],
               border=True, delta_arrow="auto" if abs(diff) >= 5e-5 else "off")

# ---------------------------------------------------------------- tabs
tab_perf, tab_cmp, tab_pred, tab_data = st.tabs([
    ":material/insights: Model performance",
    ":material/leaderboard: Compare models",
    ":material/table_rows: Predictions",
    ":material/dataset: Test data",
])

with tab_perf:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        st.caption("Cell = count and % of the actual class (rows add to 100 %).")
        show(plot_confusion(y_eval, current["pred"], CLASSES))
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(classification_report(
            y_eval, current["pred"], labels=CLASSES, output_dict=True, zero_division=0)).T
        report = report.drop(index="accuracy", errors="ignore")
        report["support"] = report["support"].astype(int)
        st.dataframe(
            report, width="stretch",
            column_config={
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1-score": st.column_config.ProgressColumn("F1", format="%.3f",
                                                            min_value=0, max_value=1),
                "support": st.column_config.NumberColumn("Students", width="small"),
            },
        )
        worst = report.loc[CLASSES, "recall"].idxmin()
        st.caption(f"Hardest class for this model: **{worst}** "
                   f"(recall {report.loc[worst, 'recall']:.2f}).")

with tab_cmp:
    best = table["F1"].idxmax()
    st.success(f"Best macro-F1 on this data: **{best}** ({table.loc[best, 'F1']:.4f}). "
               f"Highest MCC: **{table['MCC'].idxmax()}**.", icon=":material/trophy:")
    st.dataframe(
        table.style.format("{:.4f}", na_rep="n/a")
             .highlight_max(axis=0, props="font-weight:700; color:#2F9E44;"),
        width="stretch",
    )
    rank_metric = st.pills("Rank models by", list(METRIC_HELP), default="F1",
                           required=True)
    show(plot_ranking(table, rank_metric, MODEL_LABELS[chosen_key]))
    st.caption("Highlighted bar = model selected in Step 2.")

with tab_pred:
    proba = current["proba"]
    preds = pd.DataFrame({
        "Actual": y_eval,
        "Predicted": current["pred"],
        "Correct": y_eval.to_numpy() == current["pred"],
        "Confidence": proba.max(axis=1),
    }, index=X_eval.index).join(proba.add_prefix("P(").add_suffix(")"))

    f1c, f2c = st.columns([1, 2])
    show_which = f1c.segmented_control("Rows", ["All", "Correct", "Wrong"], default="All",
                                       required=True)
    class_filter = f2c.pills("Actual class", CLASSES, selection_mode="multi", default=CLASSES)

    view = preds[preds["Actual"].isin(class_filter or CLASSES)]
    if show_which == "Correct":
        view = view[view["Correct"]]
    elif show_which == "Wrong":
        view = view[~view["Correct"]]

    st.caption(f"Showing {len(view)} of {len(preds)} rows · "
               f"{(~preds['Correct']).sum()} misclassified overall.")
    st.dataframe(
        view, width="stretch", height=420,
        column_config={
            "Correct": st.column_config.CheckboxColumn("✓"),
            "Confidence": st.column_config.ProgressColumn("Confidence", format="%.2f",
                                                          min_value=0, max_value=1),
            **{f"P({c})": st.column_config.NumberColumn(f"P({c})", format="%.3f")
               for c in proba.columns},
        },
    )
    st.download_button("Download predictions", preds.to_csv(index_label="row").encode(),
                       file_name=f"predictions_{chosen_key}.csv", mime="text/csv",
                       icon=":material/download:")

with tab_data:
    d_left, d_right = st.columns([1, 1.4], gap="large")
    with d_left:
        st.markdown("**Outcome distribution**")
        show(plot_class_counts(y_eval, CLASSES))
        st.caption("Classes are imbalanced, which is why the macro-averaged metrics and "
                   "MCC matter more than accuracy alone.")
    with d_right:
        st.markdown(f"**Preview** · first 50 of {len(data)} rows")
        st.dataframe(data.head(50), width="stretch", height=300)
