import json
import sys
import matplotlib.pyplot as plt
import numpy as np

# ── Load JSON ─────────────────────────────────────────────────────────────────

json_path = sys.argv[1] if len(sys.argv) > 1 else "trainer_state.json"

with open(json_path) as f:
    state = json.load(f)

log_history = state["log_history"]

# ── Parse eval and train entries ──────────────────────────────────────────────

eval_data  = [e for e in log_history if "eval_loss" in e]
train_data = [e for e in log_history if "loss" in e and "eval_loss" not in e]

eval_epochs = [round(d["epoch"])        for d in eval_data]
eval_jga    = [d["eval_tlb_jga"] * 100  for d in eval_data]
eval_loss   = [d["eval_loss"]           for d in eval_data]

train_steps = [d["step"]          for d in train_data]
train_lr    = [d["learning_rate"] for d in train_data]
train_grad  = [d["grad_norm"]     for d in train_data]

# Derive best epoch from best_global_step in state
best_step  = state["best_global_step"]
best_epoch = eval_epochs[eval_jga.index(max(eval_jga))]  # fallback
for d in eval_data:
    if d.get("step") == best_step:
        best_epoch = round(d["epoch"])
        break

num_epochs = state.get("num_train_epochs", max(eval_epochs))

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = {
    "jga":  "#3266ad",
    "loss": "#d9534f",
    "lr":   "#5cb85c",
    "grad": "#f0a500",
    "best": "#888",
}
FIG_SIZE   = (10, 6)
FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK  = 10
FONT_ANNOT = 10

base_path = json_path.replace(".json", "")

def _best_vline(ax, x):
    ax.axvline(x, color=COLORS["best"], linestyle="--", linewidth=1.4,
               alpha=0.7, label=f"Best epoch ({x})")

def _save(fig, suffix):
    path = f"{base_path}_{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {path}")

# ── Figure 1: JGA vs epoch ────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
ax1.plot(eval_epochs, eval_jga, marker="o", color=COLORS["jga"],
         linewidth=2.5, markersize=8, zorder=3)
ax1.fill_between(eval_epochs, eval_jga, alpha=0.08, color=COLORS["jga"])
best_jga = max(eval_jga)
ax1.annotate(f"{best_jga:.1f}%",
             xy=(best_epoch, best_jga),
             xytext=(best_epoch - 1.8, best_jga - 5),
             arrowprops=dict(arrowstyle="->", color=COLORS["best"]),
             fontsize=FONT_ANNOT, color=COLORS["best"])
_best_vline(ax1, best_epoch)
ax1.set_title("Joint Goal Accuracy vs Epoch", fontsize=FONT_TITLE)
ax1.set_xlabel("Epoch", fontsize=FONT_LABEL)
ax1.set_ylabel("JGA (%)", fontsize=FONT_LABEL)
ax1.set_xticks(range(1, num_epochs + 1))
ax1.tick_params(labelsize=FONT_TICK)
ax1.set_ylim(max(0, min(eval_jga) - 10), min(100, max(eval_jga) + 10))
ax1.legend(fontsize=FONT_TICK)
ax1.grid(axis="y", linestyle="--", alpha=0.4)
fig1.tight_layout()
_save(fig1, "jga")

# ── Figure 2: Eval loss vs epoch ──────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
ax2.plot(eval_epochs, eval_loss, marker="s", color=COLORS["loss"],
         linewidth=2.5, markersize=8, zorder=3)
ax2.fill_between(eval_epochs, eval_loss, alpha=0.08, color=COLORS["loss"])
best_loss = min(eval_loss)
ax2.annotate(f"{best_loss:.4f}",
             xy=(best_epoch, best_loss),
             xytext=(best_epoch - 2.2, best_loss + max(eval_loss) * 0.07),
             arrowprops=dict(arrowstyle="->", color=COLORS["best"]),
             fontsize=FONT_ANNOT, color=COLORS["best"])
_best_vline(ax2, best_epoch)
ax2.set_title("Eval loss vs Epoch", fontsize=FONT_TITLE)
ax2.set_xlabel("Epoch", fontsize=FONT_LABEL)
ax2.set_ylabel("Eval loss", fontsize=FONT_LABEL)
ax2.set_xticks(range(1, num_epochs + 1))
ax2.tick_params(labelsize=FONT_TICK)
ax2.legend(fontsize=FONT_TICK)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
fig2.tight_layout()
_save(fig2, "eval_loss")

# ── Figure 3: Learning rate vs step ───────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=FIG_SIZE)
ax3.plot(train_steps, train_lr, color=COLORS["lr"], linewidth=2.5)
warmup_end = train_steps[train_lr.index(max(train_lr))]
ax3.axvspan(0, warmup_end, alpha=0.08, color=COLORS["lr"],
            label=f"Warmup (step 0-{warmup_end})")
ax3.set_title("Learning rate vs Training Step", fontsize=FONT_TITLE)
ax3.set_xlabel("Training step", fontsize=FONT_LABEL)
ax3.set_ylabel("Learning rate", fontsize=FONT_LABEL)
ax3.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax3.tick_params(labelsize=FONT_TICK)
ax3.legend(fontsize=FONT_TICK)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
fig3.tight_layout()
_save(fig3, "learning_rate")

# ── Figure 4: Grad norm vs step ───────────────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=FIG_SIZE)
ax4.plot(train_steps, train_grad, color=COLORS["grad"], linewidth=2,
         marker=".", markersize=6, zorder=3)
mean_grad = np.mean(train_grad)
ax4.axhline(mean_grad, color=COLORS["best"], linestyle="--",
            linewidth=1.4, label=f"Mean: {mean_grad:.2f}")
ax4.set_title("Gradient norm vs Training Step", fontsize=FONT_TITLE)
ax4.set_xlabel("Training step", fontsize=FONT_LABEL)
ax4.set_ylabel("Gradient norm", fontsize=FONT_LABEL)
ax4.tick_params(labelsize=FONT_TICK)
ax4.legend(fontsize=FONT_TICK)
ax4.grid(axis="y", linestyle="--", alpha=0.4)
fig4.tight_layout()
_save(fig4, "grad_norm")

plt.show()
