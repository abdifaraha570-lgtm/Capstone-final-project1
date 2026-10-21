"""Bias detection suite - representation bias in the data (before any model exists).

Checks, per monitored group (gender, age band, scholarship, disability):
  1. Representation: is any group under MIN_GROUP_SHARE of the records?
  2. Label balance: does the no-show rate differ by more than MAX_RATE_GAP?
  3. Split drift: is each group represented similarly in train / validation / test?
Uses Fairlearn's MetricFrame when installed; falls back to pandas with identical output.
"""
import json

import pandas as pd

from src import config
from src.utils.audit_log import write_event


def group_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    y = df["no_show"]
    try:
        from fairlearn.metrics import MetricFrame, count, selection_rate
        mf = MetricFrame(metrics={"records": count, "no_show_rate": selection_rate},
                         y_true=y, y_pred=y, sensitive_features=df[group])
        t = mf.by_group.reset_index().rename(columns={group: "group"})
    except ImportError:
        t = (df.groupby(group)["no_show"].agg(records="size", no_show_rate="mean")
               .reset_index().rename(columns={group: "group"}))
    t["share"] = t["records"] / t["records"].sum()
    t.insert(0, "attribute", group)
    return t


def run(df: pd.DataFrame, splits: dict | None = None) -> dict:
    tables, flags = [], []
    for g in config.BIAS_GROUPS:
        t = group_table(df, g)
        tables.append(t)
        for _, r in t[t["share"] < config.MIN_GROUP_SHARE].iterrows():
            flags.append({"attribute": g, "group": str(r["group"]), "issue": "under-represented",
                          "share": round(float(r["share"]), 4)})
        gap = float(t["no_show_rate"].max() - t["no_show_rate"].min())
        if gap > config.MAX_RATE_GAP:
            flags.append({"attribute": g, "issue": "label-rate gap", "gap": round(gap, 4)})

    drift = []
    if splits:
        for g in config.BIAS_GROUPS:
            shares = {n: s[g].value_counts(normalize=True) for n, s in splits.items()}
            frame = pd.DataFrame(shares).fillna(0)
            max_diff = float((frame.max(axis=1) - frame.min(axis=1)).max())
            drift.append({"attribute": g, "max_share_diff_across_splits": round(max_diff, 4)})
            if max_diff > 0.02:
                flags.append({"attribute": g, "issue": "split drift", "diff": round(max_diff, 4)})

    report = pd.concat(tables, ignore_index=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.round(4).to_csv(config.REPORTS_DIR / "bias_representation.csv", index=False)
    result = {"flags": flags, "split_drift": drift,
              "note": "Flags trigger human review by the AI & Data Governance Committee; "
                      "they do not delete groups or auto-reweight data."}
    (config.REPORTS_DIR / "bias_flags.json").write_text(json.dumps(result, indent=2))
    write_event("VALIDATE", "appointments_features", "representation bias check",
                flags=len(flags))
    _plot(report)
    return result


def _plot(report: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(config.BIAS_GROUPS), figsize=(14, 3.6))
    for ax, g in zip(axes, config.BIAS_GROUPS):
        t = report[report["attribute"] == g]
        ax.bar(t["group"].astype(str), t["no_show_rate"], color="#1F6F78")
        ax.axhline(report[report["attribute"] == g]["no_show_rate"].mean(), ls="--", c="#C8553D", lw=1)
        ax.set_title(g); ax.set_ylim(0, 0.4); ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("no-show rate")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "bias_representation.png", dpi=160)
    plt.close(fig)
