import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Initialize 16:9 widescreen presentation
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)

# Define Colors
C_WHITE = RGBColor(255, 255, 255)
C_BLACK = RGBColor(24, 24, 27)
C_DARK_CARD = RGBColor(20, 20, 23)
C_ORANGE_LPU = RGBColor(227, 90, 31)
C_GRAY_BG = RGBColor(248, 250, 252)
C_GRAY_BORDER = RGBColor(226, 232, 240)
C_TEXT_DARK = RGBColor(15, 23, 42)
C_TEXT_MUTED = RGBColor(100, 116, 139)
C_BLUE = RGBColor(37, 99, 235)
C_EMERALD = RGBColor(16, 185, 129)
C_AMBER = RGBColor(245, 158, 11)

FONT_NAME = "Arial"

# Background fill
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = C_WHITE
bg.line.fill.background()

# -------------------------------------------------------------
# 1. HEADER (LPU Logo, Orange Rule, NAAC A++, Slide Title)
# -------------------------------------------------------------
header_path = Path("artifacts/lpu_slide_header.png")
if header_path.exists():
    slide.shapes.add_picture(str(header_path), 0, 0, width=prs.slide_width, height=Inches(0.65))
else:
    # Fallback header
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.55), prs.slide_width, Inches(0.04))
    rule.fill.solid()
    rule.fill.fore_color.rgb = C_ORANGE_LPU
    rule.line.fill.background()

# Slide Title
title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.72), Inches(12.1), Inches(0.65))
tf = title_box.text_frame
tf.word_wrap = True
tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
p = tf.paragraphs[0]
p.text = "Model Selection - Prediction part"
p.font.name = FONT_NAME
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = C_TEXT_DARK

# -------------------------------------------------------------
# 2. TOP ROW: Visual Evaluations & Metrics Card
# -------------------------------------------------------------
# Plot 1: Model Comparison (MAE / RMSE & Surge Inflection)
eval_curves_path = Path("artifacts/prediction_models_eval_curves.png")
if eval_curves_path.exists():
    slide.shapes.add_picture(str(eval_curves_path), Inches(0.6), Inches(1.42), width=Inches(4.6), height=Inches(2.35))

# Plot 2: Multi-Horizon Progression
horizon_path = Path("artifacts/prediction_multi_horizon_plot.png")
if horizon_path.exists():
    slide.shapes.add_picture(str(horizon_path), Inches(5.35), Inches(1.42), width=Inches(2.45), height=Inches(2.35))

# Card 3: Metrics Box (Matching RF-DETR Card Style)
card_metric = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.95), Inches(1.42), Inches(4.75), Inches(2.35))
card_metric.fill.solid()
card_metric.fill.fore_color.rgb = C_GRAY_BG
card_metric.line.color.rgb = C_GRAY_BORDER
card_metric.line.width = Pt(1.2)

tb_m = slide.shapes.add_textbox(Inches(8.15), Inches(1.5), Inches(4.35), Inches(2.2))
tf_m = tb_m.text_frame
tf_m.word_wrap = True
tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = 0

p = tf_m.paragraphs[0]
p.text = "Metrics  ⓘ"
p.font.name = FONT_NAME
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = C_TEXT_DARK

p_sub = tf_m.add_paragraph()
p_sub.text = "Holdout Test Set"
p_sub.font.name = FONT_NAME
p_sub.font.size = Pt(9.5)
p_sub.font.color.rgb = C_TEXT_MUTED
p_sub.space_after = Pt(2)

p_model = tf_m.add_paragraph()
p_model.text = "Tuned XGBoost Forecaster (+15m) & Dual-Head"
p_model.font.name = FONT_NAME
p_model.font.size = Pt(13)
p_model.font.bold = True
p_model.font.color.rgb = C_TEXT_DARK
p_model.space_after = Pt(8)

# Metric Pills Grid inside the card
metrics_data = [
    ("R² Score", "0.9905", C_BLUE),
    ("MAE (+15m)", "4.88 cars", C_EMERALD),
    ("Surge MAE", "9.84 cars", C_ORANGE_LPU),
    ("PR-AUC", "0.9498", C_BLUE)
]

for label, val, color in metrics_data:
    p_row = tf_m.add_paragraph()
    p_row.text = f"• {label}:  {val}"
    p_row.font.name = FONT_NAME
    p_row.font.size = Pt(10.5)
    p_row.font.bold = True
    p_row.font.color.rgb = color
    p_row.space_after = Pt(2)

# -------------------------------------------------------------
# 3. BOTTOM ROW: Feature Pipeline Card, Trace Visual & Table
# -------------------------------------------------------------
# Bottom Left: Dark Card (Feature Pipeline & Split Rigor)
dark_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(3.92), Inches(3.7), Inches(3.2))
dark_card.fill.solid()
dark_card.fill.fore_color.rgb = C_DARK_CARD
dark_card.line.fill.background()

tb_dark = slide.shapes.add_textbox(Inches(0.75), Inches(4.02), Inches(3.4), Inches(3.0))
tf_d = tb_dark.text_frame
tf_d.word_wrap = True
tf_d.margin_left = tf_d.margin_top = tf_d.margin_right = tf_d.margin_bottom = 0

p = tf_d.paragraphs[0]
p.text = "Feature Pipeline & Split Rigor:"
p.font.name = FONT_NAME
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = C_WHITE
p.space_after = Pt(4)

bullets = [
    "Chronological Split: 70% Train, 15% Val, 15% Test",
    "Purge Buffer: 1-hour temporal gap (zero leakage)",
    "Cyclical Encodings: sin/cos diurnal & weekly curves",
    "Causal Auto-Lags: y(t-15m), y(t-30m), y(t-45m), y(t-60m)",
    "Demand Velocity: Δ_15m = y_t - y_t-15, Δ_30m = y_t - y_t-30",
    "Shifted Rolling Windows: 30m & 60m from t-1 backward",
    "Normal-Day Profile: Leave-one-week-out zone baseline",
    "Calendar Lookaheads: Bell window, exam & event flags",
    "Groupby Isolation: Eliminates cross-zone lag bleed"
]

for b in bullets:
    p_b = tf_d.add_paragraph()
    p_b.text = f"•  {b}"
    p_b.font.name = FONT_NAME
    p_b.font.size = Pt(8.8)
    p_b.font.color.rgb = RGBColor(228, 228, 231)
    p_b.space_after = Pt(2)

# Bottom Middle: Rush-Hour Surge Trace Graphic
trace_path = Path("artifacts/rush_surge_trace_plot.png")
if trace_path.exists():
    slide.shapes.add_picture(str(trace_path), Inches(4.45), Inches(3.92), width=Inches(3.35), height=Inches(3.2))

# Bottom Right: Table (Component | Optimisation Technique & Selection)
tbl_shape = slide.shapes.add_table(6, 2, Inches(7.95), Inches(3.92), Inches(4.75), Inches(3.2))
tbl = tbl_shape.table
tbl.columns[0].width = Inches(2.2)
tbl.columns[1].width = Inches(2.55)

table_data = [
    ("Component", "Optimisation technique"),
    ("YOLO vehicle detection", "AdamW"),
    ("XGBoost parking forecasting", "GridSearchCV + TimeSeriesSplit"),
    ("XGBoost internal training", "Gradient boosting with regularisation"),
    ("Saturation classifier", "Depends on its underlying implementation"),
    ("Model selection verdict", "Tuned XGBoost chosen (<3ms latency, 40% surge gain)")
]

for row_idx, row in enumerate(table_data):
    is_hdr = (row_idx == 0)
    is_verdict = (row_idx == 5)
    for col_idx, text in enumerate(row):
        cell = tbl.cell(row_idx, col_idx)
        cell.fill.solid()
        if is_hdr:
            cell.fill.fore_color.rgb = C_WHITE
        elif is_verdict:
            cell.fill.fore_color.rgb = RGBColor(238, 242, 255) # soft indigo
        else:
            cell.fill.fore_color.rgb = C_WHITE
        
        p = cell.text_frame.paragraphs[0]
        p.text = text
        p.font.name = FONT_NAME
        p.font.size = Pt(9.5 if not is_hdr else 10.5)
        p.font.bold = is_hdr or (is_verdict and col_idx == 0)
        p.font.color.rgb = C_TEXT_DARK if not is_verdict else RGBColor(30, 58, 138)

# -------------------------------------------------------------
# 4. FOOTER: Orange Bottom Accent Bar
# -------------------------------------------------------------
footer_path = Path("artifacts/lpu_slide_footer.png")
if footer_path.exists():
    slide.shapes.add_picture(str(footer_path), 0, Inches(7.33), width=prs.slide_width, height=Inches(0.17))
else:
    footer = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.33), prs.slide_width, Inches(0.17))
    footer.fill.solid()
    footer.fill.fore_color.rgb = C_ORANGE_LPU
    footer.line.fill.background()

# Speaker notes
slide.notes_slide.notes_text_frame.text = (
    "SPEAKER SCRIPT FOR 'MODEL SELECTION - PREDICTION PART':\n"
    "Respected examiners, while our detection model used AdamW to fine-tune YOLO, our prediction stage "
    "evaluates and selects the optimal machine learning model for time-series forecasting.\n"
    "Referencing Notebook 02, we benchmarked Persistence, Seasonal Historical Mean, Ridge Regression, Random Forest, "
    "and Tuned XGBoost. During calm hours, naive persistence achieves an MAE of 6.6 cars, but during rush-hour lecture "
    "transitions, its error explodes to 15.3 cars. Tuned XGBoost cuts this rush surge error down to 9.8 cars—a 40% error reduction.\n"
    "Hyperparameters were optimized via GridSearchCV across a 5-fold TimeSeriesSplit with a 1-hour temporal purge buffer, "
    "yielding an R-squared of 0.9905 with sub-3 millisecond inference latency. Combined with our dual-head saturation classifier "
    "achieving 0.9498 PR-AUC, Tuned XGBoost was conclusively selected for deployment."
)

output_pptx = Path("artifacts/Model_Selection_Prediction_Slide.pptx")
prs.save(str(output_pptx))
print(f"[SUCCESS] Standalone slide saved to {output_pptx.resolve()}")
