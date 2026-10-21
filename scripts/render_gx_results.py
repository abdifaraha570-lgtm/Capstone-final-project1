"""Render the saved GX checkpoint results as a compact image (used if Data Docs screenshots fail,
and useful as a readable summary alongside them)."""
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def render(suite: str, out: str) -> None:
    path = ROOT / "reports" / f"gx_{suite}.json"
    if not path.exists():
        print("missing", path); return
    data = json.loads(path.read_text())
    groups = OrderedDict()
    for r in data.get("details", []):
        g = groups.setdefault(str(r["expectation"]).replace("expect_", ""), {"cols": [], "ok": 0, "n": 0, "bad": 0})
        g["cols"].append(str(r["column"])); g["n"] += 1; g["ok"] += int(r["success"])
        g["bad"] += int(r.get("unexpected_count") or 0)
    rows = []
    for name, g in groups.items():
        cols = ", ".join(g["cols"]) if len(g["cols"]) <= 3 else f"{len(g['cols'])} columns"
        rows.append([name, cols, f"{g['ok']}/{g['n']}", "PASS" if g["ok"] == g["n"] else "FAIL",
                     f"{g['bad']:,}" if g["bad"] else "-"])
    fig, ax = plt.subplots(figsize=(10, 0.9 + 0.3 * len(rows))); ax.axis("off")
    total_ok = sum(int(r["success"]) for r in data.get("details", []))
    engine = "pandas validator" if data.get("engine") == "pandas" else "Great Expectations"
    ax.set_title(f"{suite} ({engine})  -  {total_ok}/{len(data.get('details', []))} passed  -  "
                 f"success: {data['success']}", loc="left", fontsize=12, fontweight="bold", color="#0E3B43")
    t = ax.table(cellText=rows, colLabels=["Expectation", "Column(s)", "Passed", "Result", "Unexpected rows"],
                 loc="upper left", colWidths=[0.40, 0.27, 0.09, 0.09, 0.15], cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#B9CFD2")
        if r == 0:
            cell.set_facecolor("#1F6F78"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
        elif c == 3:
            cell.set_facecolor("#DDF0E0" if rows[r - 1][3] == "PASS" else "#F6D5D2")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    render("raw_appointments_suite", "gx_raw_results.png")
    render("processed_appointments_suite", "gx_processed_results.png")
