import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Create figure matching slide aspect ratio (1920x1080)
fig = plt.figure(figsize=(16, 9), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Background
bg = patches.Rectangle((0, 0), 16, 9, facecolor='#FFFFFF')
ax.add_patch(bg)

# 1. Header (LPU logo, rule, NAAC A++)
header_img_path = 'artifacts/lpu_slide_header.png'
if os.path.exists(header_img_path):
    h_img = Image.open(header_img_path)
    ax.imshow(h_img, extent=[0, 16, 8.25, 9.0], aspect='auto', zorder=2)
else:
    rule = patches.Rectangle((0, 8.4), 16, 0.05, facecolor='#E35A1F', zorder=2)
    ax.add_patch(rule)

# Slide Title
ax.text(0.7, 7.6, "Model Selection - Prediction part", fontsize=25, fontweight='bold', color='#0F172A', zorder=3)

# 2. Top Row
# Image 1: Model Comparison Curves (MAE / RMSE & Surge)
p1_path = 'artifacts/prediction_models_eval_curves.png'
if os.path.exists(p1_path):
    p1 = Image.open(p1_path)
    ax.imshow(p1, extent=[0.7, 6.2, 4.75, 7.35], aspect='auto', zorder=2)

# Image 2: Multi-Horizon Degradation
p2_path = 'artifacts/prediction_multi_horizon_plot.png'
if os.path.exists(p2_path):
    p2 = Image.open(p2_path)
    ax.imshow(p2, extent=[6.4, 9.6, 4.75, 7.35], aspect='auto', zorder=2)

# Card 3: Metrics Box (Matching RF-DETR Style)
card_metric = patches.FancyBboxPatch((9.8, 4.75), 5.5, 2.6,
                                     boxstyle="round,pad=0.15",
                                     edgecolor='#E2E8F0', facecolor='#F8FAFC', lw=1.5, zorder=2)
ax.add_patch(card_metric)

ax.text(10.1, 7.05, "Metrics  ⓘ", fontsize=12, fontweight='bold', color='#0F172A', zorder=3)
ax.text(10.1, 6.75, "Holdout Test Set", fontsize=10, color='#64748B', zorder=3)
ax.text(10.1, 6.35, "Tuned XGBoost Forecaster (+15m) & Dual-Head", fontsize=12.5, fontweight='bold', color='#0F172A', zorder=3)

# Metrics Grid inside card
ax.text(10.1, 5.85, "• R² Score:              0.9905", fontsize=11, fontweight='bold', color='#2563EB', zorder=3)
ax.text(10.1, 5.50, "• MAE (+15m):         4.88 cars", fontsize=11, fontweight='bold', color='#10B981', zorder=3)
ax.text(10.1, 5.15, "• Surge Inflection:   9.84 cars (~40% gain)", fontsize=11, fontweight='bold', color='#E35A1F', zorder=3)
ax.text(10.1, 4.80, "• Saturation PR-AUC: 0.9498 (Recall: 97.5%)", fontsize=11, fontweight='bold', color='#2563EB', zorder=3)

# 3. Bottom Row
# Bottom Left: Dark Card (Feature Pipeline & Split Rigor)
dark_card = patches.FancyBboxPatch((0.7, 0.4), 4.4, 4.0,
                                   boxstyle="round,pad=0.15",
                                   edgecolor='none', facecolor='#141417', zorder=2)
ax.add_patch(dark_card)

ax.text(0.9, 4.05, "Feature Pipeline & Split Rigor:", fontsize=12, fontweight='bold', color='#FFFFFF', zorder=3)

bullets = [
    "Chronological Split: 70% Train, 15% Val, 15% Test",
    "Purge Buffer: 1-hour temporal gap (zero leakage)",
    "Cyclical Encodings: sin/cos diurnal & weekly curves",
    "Causal Auto-Lags: y(t-15m), y(t-30m), y(t-45m), y(t-60m)",
    "Demand Velocity: Δ_15m = y_t - y_t-15, Δ_30m = y_t - y_t-30",
    "Shifted Rolling Stats: 30m & 60m from t-1 backward",
    "Normal-Day Profile: Leave-one-week-out zone baseline",
    "Calendar Lookaheads: Bell window, exam & event flags",
    "Groupby Isolation: Eliminates cross-zone lag bleed"
]

y_pos = 3.65
for b in bullets:
    ax.text(0.9, y_pos, f"•  {b}", fontsize=9.2, color='#E4E4E7', zorder=3)
    y_pos -= 0.35

# Bottom Middle: Rush Surge Trace Graphic
trace_path = 'artifacts/rush_surge_trace_plot.png'
if os.path.exists(trace_path):
    tr = Image.open(trace_path)
    ax.imshow(tr, extent=[5.35, 9.75, 0.35, 4.45], aspect='auto', zorder=2)

# Bottom Right: Table
# Draw table using Matplotlib table
table_data = [
    ["Component", "Optimisation technique"],
    ["YOLO vehicle detection", "AdamW"],
    ["XGBoost parking forecasting", "GridSearchCV + TimeSeriesSplit"],
    ["XGBoost internal training", "Gradient boosting with regularisation"],
    ["Saturation classifier", "Scale-pos-weight + Isotonic calibration"],
    ["Model selection verdict", "Tuned XGBoost chosen (<3ms, 40% gain)"]
]

table_ax = fig.add_axes([0.625, 0.05, 0.35, 0.44])
table_ax.axis('off')
tbl = table_ax.table(cellText=table_data, loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.0, 1.8)

# Style table cells
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#FFFFFF')
        cell.set_text_props(weight='bold', color='#0F172A')
    elif r == 5:
        cell.set_facecolor('#EEF2FF')
        cell.set_text_props(weight='bold', color='#1E3A8A')
    else:
        cell.set_facecolor('#FFFFFF')
        cell.set_text_props(color='#1E293B')
    cell.set_edgecolor('#CBD5E1')

# 4. Footer (Orange Bar)
footer_img_path = 'artifacts/lpu_slide_footer.png'
if os.path.exists(footer_img_path):
    f_img = Image.open(footer_img_path)
    ax.imshow(f_img, extent=[0, 16, 0, 0.22], aspect='auto', zorder=4)
else:
    footer = patches.Rectangle((0, 0), 16, 0.2, facecolor='#E35A1F', zorder=4)
    ax.add_patch(footer)

output_img = 'artifacts/model_selection_prediction_slide.png'
plt.savefig(output_img, dpi=200, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"[SUCCESS] Rendered slide image saved to {output_img}")
