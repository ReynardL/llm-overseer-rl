"""Draw figures/results.png from results/evals.json.   python results/plot.py   (needs matplotlib)"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "evals.json").read_text())
J = [json.loads(l) for l in (HERE / "journal.jsonl").read_text().splitlines()]

BLUE, FAINT, ORANGE, GREY = "#257de0", "#a8cbf3", "#f47230", "#9a9a9a"
INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 12, "text.color": INK,
                     "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=200, gridspec_kw={"width_ratios": [3.6, 1], "wspace": 0.16})
fig.subplots_adjust(left=0.065, right=0.975, top=0.80, bottom=0.17)
fig.text(0.065, 0.945, "An LLM overseer against fixed settings", fontsize=19, weight="bold")
fig.text(0.065, 0.895, "Qwen3-1.7B learning to fix Python bugs with tools, trained with GRPO on one RTX 4060 (8 GB). "
         "52,000 rollouts each.", fontsize=12.5, color=INK2)

# left: validation pass rate at every eval
main, a2 = D["arms"]["main"], D["arms"]["a2s1"]
xs = lambda rows: [r["rollouts"] for r in rows]
ys = lambda rows: [r["val"] * 100 for r in rows]
for x in D["budget_extensions"]:
    ax.axvline(J[x["before_decision"] - 1]["rollouts_used"], color=AXIS, lw=1, zorder=1)
first = J[D["budget_extensions"][0]["before_decision"] - 1]["rollouts_used"]
ax.text(first - 350, 2.5, "grey lines: I extended the\noverseer's budget by hand", fontsize=10, color=INK2, va="bottom", ha="right")
ax.plot(xs(a2), ys(a2), color=ORANGE, lw=2, solid_capstyle="round", solid_joinstyle="round", label="Fixed settings", zorder=3)
val = {r["eval"]: r for r in main}
pt = lambda e: (val[e]["rollouts"], val[e]["val"] * 100)
start = lambda j: j["target"] if j["decision"] == "REVERT" else j["eval"]  # the checkpoint the next 800 rollouts train from
parent = {"e1": "e0"} | {f"e{int(j['eval'][1:]) + 1}": start(j) for j in J}
kept, e = [], D["picks"]["main"]  # the picked checkpoint and every eval it descends from
while e:
    kept.append(e)
    e = parent.get(e)
ax.plot(*zip(*map(pt, kept[::-1])), color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round",
        label="LLM overseer, training behind the selected checkpoint", zorder=4)
for e in val:  # training the overseer dropped by going back: a faint stub from the checkpoint it started at
    if e not in kept:
        ax.plot(*zip(pt(parent[e]), pt(e)), color=FAINT, lw=1.4, marker="o", ms=3.5, markevery=[1], solid_capstyle="round",
                zorder=2, label="LLM overseer, training it discarded by going back" if e == "e13" else None)
ch = [start(j) for j in J if j["changed"] and start(j) in kept]
ax.plot(*zip(*map(pt, ch)), "o", ms=6.5, color=BLUE, mec="white", mew=1.6, zorder=5, label="Overseer changed the settings here")
for rows, arm in ((main, "main"), (a2, "a2s1")):
    r = next(r for r in rows if r["eval"] == D["picks"][arm])
    ax.plot(r["rollouts"], r["val"] * 100, "o", ms=9, color=BLUE if arm == "main" else ORANGE, mec="white", mew=2, zorder=6)
    ax.annotate(f"{r['val'] * 100:.0f}%", (r["rollouts"], r["val"] * 100), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=12.5, weight="bold")


def note(i, text, x, y, ha="left", at=None):  # text at data position (x, y), leader to where decision i trains from
    ax.annotate(text, pt(at or start(J[i - 1])), xytext=(x, y), textcoords="data", ha=ha, va="center",
                fontsize=10.5, color=INK, arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 1, "shrinkB": 5}, zorder=7)


note(2, "raises learning rate,\ndoubles group size", 3300, 14.5)
note(10, "adds harder tasks", 5600, 50, "center")
note(21, "stalls near 45% and keeps\nlowering the learning rate", 12600, 57, "center")
note(27, "tries a learning rate\n5x the starting one", 21200, 69, "right")
note(45, "two tries collapse and are discarded,\nthen the rate is lowered", 37300, 57, at="e45")
ax.set_xlim(-600, 53500)
ax.set_ylim(0, 85)
ax.set_xticks(range(0, 50001, 10000), [f"{k}K" if k else "0" for k in range(0, 51, 10)])
ax.set_yticks(range(0, 81, 20), [f"{k}%" for k in range(0, 81, 20)])
ax.set_xlabel("Training rollouts used, discarded training included")
ax.set_title(f"Validation pass rate ({D['n_val']} tasks), scored every 800 rollouts", fontsize=12.5, color=INK, loc="left", pad=10)
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
ax.legend(loc="lower right", fontsize=10.5, handlelength=1.6, labelspacing=0.3, borderpad=0.2, borderaxespad=0.3, facecolor="white", edgecolor="none", framealpha=1)

# right: test, read once per picked checkpoint
T = D["test"]
bars = [("Before\ntraining", T["main_e0"], GREY), ("Fixed\nsettings", T[f"a2s1_{D['picks']['a2s1']}"], ORANGE),
        ("LLM\noverseer", T[f"main_{D['picks']['main']}"], BLUE)]
for i, (name, t, c) in enumerate(bars):
    bx.bar(i, t["pass_rate"] * 100, width=0.42, color=c, zorder=3)
    bx.text(i, t["pass_rate"] * 100 + 1.6, f"{t['pass_rate'] * 100:.0f}%", ha="center", fontsize=14, weight="bold")
bx.set_xticks(range(3), [b[0] for b in bars], color=INK)
bx.set_xlim(-0.6, 2.6)
bx.set_ylim(0, 85)
bx.set_yticks(range(0, 81, 20), [""] * 5)
bx.grid(axis="y", color=GRID, lw=1)
bx.set_axisbelow(True)
bx.set_title(f"Held-out test ({T['main_e0']['n']} tasks)", fontsize=12.5, color=INK, loc="left", pad=10)
d = T["main_minus_a2s1"]
bx.set_xlabel(f"Overseer minus fixed: {d['delta'] * 100:.0f} points\n(standard error {d['se'] * 100:.1f})", color=INK2, labelpad=8)
for a in (ax, bx):
    a.spines[["top", "right", "left"]].set_visible(False)
    a.tick_params(length=0)

out = HERE.parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "results.png")
print("wrote", out / "results.png")
