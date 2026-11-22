import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# COLOR PALETTE (Modern Academic / Executive Tech)
# ---------------------------------------------------------------------------
C_NAVY_DARK   = RGBColor(15, 23, 42)      # #0F172A - Main dark background / dark text
C_NAVY_SLATE  = RGBColor(30, 41, 59)      # #1E293B - Card header / dark accents
C_PRIMARY_BLUE= RGBColor(37, 99, 235)     # #2563EB - Brand Blue
C_ACCENT_CYAN = RGBColor(14, 165, 233)    # #0EA5E9 - Sky Blue
C_EMERALD     = RGBColor(16, 185, 129)    # #10B981 - Success Green
C_AMBER       = RGBColor(245, 158, 11)    # #F59E0B - Warning Amber
C_ROSE        = RGBColor(239, 68, 68)     # #EF4444 - Alert Red
C_BG_LIGHT    = RGBColor(248, 250, 252)   # #F8FAFC - Canvas Light Gray
C_WHITE       = RGBColor(255, 255, 255)   # #FFFFFF - Pure White
C_BORDER      = RGBColor(226, 232, 240)   # #E2E8F0 - Subtle Card Border
C_TEXT_MUTED  = RGBColor(100, 116, 139)   # #64748B - Secondary Gray
C_TEXT_DARK   = RGBColor(15, 23, 42)      # #0F172A - Primary Dark Text
C_PILL_BG     = RGBColor(219, 234, 254)   # #DBEAFE - Pill light blue background

FONT_FAMILY   = "Calibri"

def set_slide_bg(slide, prs, color):
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = color
    bg_shape.line.fill.background()
    return bg_shape

def add_header(slide, tag_text, title_text, subtitle_text=""):
    # Category Tag Pill
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(4.5), Inches(0.35))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = tag_text.upper()
    p_tag.font.name = FONT_FAMILY
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_PRIMARY_BLUE

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.55))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_FAMILY
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = C_NAVY_DARK

    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.7), Inches(0.35))
        tf_s = sub_box.text_frame
        tf_s.word_wrap = True
        tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.name = FONT_FAMILY
        p_s.font.size = Pt(12)
        p_s.font.color.rgb = C_TEXT_MUTED

def add_card(slide, left, top, width, height, bg_color=C_WHITE, border_color=C_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
    else:
        shape.line.fill.background()
    return shape

def add_speaker_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text

# ---------------------------------------------------------------------------
# INITIALIZE PRESENTATION (16:9 Widescreen)
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

print("[INFO] Creating 10-slide academic deck...")

# ===========================================================================
# SLIDE 1: TITLE SLIDE (Executive Dark Theme)
# ===========================================================================
s1 = prs.slides.add_slide(blank_layout)
set_slide_bg(s1, prs, C_NAVY_DARK)

# Decorative gradient strip at top
strip = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
strip.fill.solid()
strip.fill.fore_color.rgb = C_PRIMARY_BLUE
strip.line.fill.background()

# University & Project Tag
tb1_tag = s1.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.3), Inches(0.4))
p = tb1_tag.text_frame.paragraphs[0]
p.text = "LOVELY PROFESSIONAL UNIVERSITY  •  SCHOOL OF COMPUTER SCIENCE & ENGINEERING  •  PROJECT 32"
p.font.name = FONT_FAMILY
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = C_ACCENT_CYAN

# Main Title
tb1_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.45), Inches(11.3), Inches(1.5))
tf = tb1_title.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Campus Parking Occupancy Prediction\n& Intelligent Parking Assistant"
p.font.name = FONT_FAMILY
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = C_WHITE

# Subtitle
tb1_sub = s1.shapes.add_textbox(Inches(1.0), Inches(3.1), Inches(11.3), Inches(0.8))
p = tb1_sub.text_frame.paragraphs[0]
p.text = "An End-to-End Deep Learning, Multi-Horizon Time-Series Forecasting & Traffic Dispersion Microservice"
p.font.name = FONT_FAMILY
p.font.size = Pt(15)
p.font.color.rgb = RGBColor(148, 163, 184)

# Feature Badges Card
add_card(s1, Inches(1.0), Inches(3.9), Inches(11.333), Inches(1.0), bg_color=C_NAVY_SLATE, border_color=None)
tb_badges = s1.shapes.add_textbox(Inches(1.2), Inches(4.0), Inches(10.9), Inches(0.8))
tf_b = tb_badges.text_frame
p = tf_b.paragraphs[0]
p.text = "CORE SYSTEM HIGHLIGHTS"
p.font.name = FONT_FAMILY
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = C_EMERALD

p2 = tf_b.add_paragraph()
p2.text = "✔ Dual YOLO (91.6% mAP50) + WebGPU   ✔ 41 Causal Time-Series Features   ✔ Chronological Split & Optuna Tuning   ✔ Anti-Herding Dispatch"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(13)
p2.font.color.rgb = C_WHITE

# Team & Metadata Cards
# Left Card: Student Presenters
add_card(s1, Inches(1.0), Inches(5.2), Inches(5.45), Inches(1.6), bg_color=C_NAVY_SLATE, border_color=None)
tb_team = s1.shapes.add_textbox(Inches(1.25), Inches(5.35), Inches(5.0), Inches(1.3))
tf_team = tb_team.text_frame
p = tf_team.paragraphs[0]
p.text = "PROJECT AUTHORS"
p.font.name = FONT_FAMILY
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = C_ACCENT_CYAN

p = tf_team.add_paragraph()
p.text = "• Mohammed Shujath Ali (Reg No: 12404599)"
p.font.name = FONT_FAMILY
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_WHITE

p = tf_team.add_paragraph()
p.text = "  ML Architecture, TimeSeriesSplit, XAI, Dispatch & Full-Stack"
p.font.name = FONT_FAMILY
p.font.size = Pt(10)
p.font.color.rgb = RGBColor(148, 163, 184)

p = tf_team.add_paragraph()
p.text = "• SREEVYAS CS (Reg No: 12400848)"
p.font.name = FONT_FAMILY
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_WHITE

p = tf_team.add_paragraph()
p.text = "  Computer Vision, Jordan PIP Geometry, Spatial NMS & Test Suite"
p.font.name = FONT_FAMILY
p.font.size = Pt(10)
p.font.color.rgb = RGBColor(148, 163, 184)

# Right Card: Academic Supervision & Course
add_card(s1, Inches(6.883), Inches(5.2), Inches(5.45), Inches(1.6), bg_color=C_NAVY_SLATE, border_color=None)
tb_sup = s1.shapes.add_textbox(Inches(7.15), Inches(5.35), Inches(5.0), Inches(1.3))
tf_sup = tb_sup.text_frame
p = tf_sup.paragraphs[0]
p.text = "ACADEMIC SUPERVISION"
p.font.name = FONT_FAMILY
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = C_ACCENT_CYAN

p = tf_sup.add_paragraph()
p.text = "Faculty Supervisor: Dr. Mohit Arora"
p.font.name = FONT_FAMILY
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_WHITE

p = tf_sup.add_paragraph()
p.text = "Course: Machine Learning (Semester 5)"
p.font.name = FONT_FAMILY
p.font.size = Pt(12)
p.font.color.rgb = RGBColor(203, 213, 225)

p = tf_sup.add_paragraph()
p.text = "Academic Session: 2026-2027  •  Phagwara, Punjab"
p.font.name = FONT_FAMILY
p.font.size = Pt(11)
p.font.color.rgb = RGBColor(148, 163, 184)

add_speaker_notes(s1, (
    "SPEAKER SCRIPT (0:45):\n"
    "Good morning respected evaluators and supervisor. We present Project 32: an end-to-end "
    "Machine Learning, Computer Vision, and Decision Support microservice engineered specifically for "
    "the 600-acre Lovely Professional University campus.\n"
    "Our system addresses real-world unpaved parking lots using camera-based edge vision, "
    "a zero-leakage 41-feature time-series pipeline, calibrated prediction ranges, and an anti-herding "
    "recommendation engine."
))

# ===========================================================================
# SLIDE 2: PROBLEM STATEMENT & TOPOLOGY
# ===========================================================================
s2 = prs.slides.add_slide(blank_layout)
set_slide_bg(s2, prs, C_BG_LIGHT)
add_header(s2, "Module 01 : Context & Problem Statement", "Campus Dynamics & Empirical Ground Topology", "Addressing open-ground parking, lecture bell arrival spikes, and recommendation herd collapse")

# Left Column: The 3 Core Engineering Challenges
add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
tb = s2.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True

def add_bullet(tf, title, body, is_first=False):
    p = tf.paragraphs[0] if is_first else tf.add_paragraph()
    p.text = title
    p.font.name = FONT_FAMILY
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_NAVY_DARK
    p.space_after = Pt(2)
    p2 = tf.add_paragraph()
    p2.text = body
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(11)
    p2.font.color.rgb = C_TEXT_MUTED
    p2.space_after = Pt(10)

add_bullet(tf, "1. Unpaved, Open Campus Grounds", 
           "Traditional smart parking assumes marked asphalt bays with wired inductive loops. On LPU's 600-acre campus, parking takes place on gravel fields where physical sensors are unviable.", is_first=True)
add_bullet(tf, "2. Lecture Bell Arrival Shocks", 
           "50-minute academic intervals create massive arrival waves. Inflow spikes by over 300% within a narrow 10-minute window (:50 to :10 past the hour), rendering static systems useless.")
add_bullet(tf, "3. Recommendation 'Herd Collapse'", 
           "Standard navigation apps direct all incoming drivers to the single emptiest zone. This abruptly overflows the lot before drivers arrive, inducing localized campus gridlock.")

# Right Column: 12-Zone Stratification Table
add_card(s2, Inches(6.7), Inches(1.8), Inches(5.8), Inches(5.2))
tb_t = s2.shapes.add_textbox(Inches(6.9), Inches(1.95), Inches(5.4), Inches(0.4))
p = tb_t.text_frame.paragraphs[0]
p.text = "12 Monitored Campus Zones (2,200+ Bays)"
p.font.name = FONT_FAMILY
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_PRIMARY_BLUE

# Create Table for Zones
table_shape = s2.shapes.add_table(5, 3, Inches(6.9), Inches(2.4), Inches(5.4), Inches(4.3))
table = table_shape.table
table.columns[0].width = Inches(1.6)
table.columns[1].width = Inches(2.6)
table.columns[2].width = Inches(1.2)

zone_headers = ["Category", "Monitored LPU Zones", "Total Bays"]
for col_idx, h in enumerate(zone_headers):
    cell = table.cell(0, col_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = C_NAVY_SLATE
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

zone_data = [
    ("Transit & Gate", "ZONE_MAINGATE (GT Road Entry)", "240 bays"),
    ("Academic Core", "Block 1, Main Park North (CSE Block 26),\nMain Park South (Block 39), Arch, Law, Block 57", "1,295 bays"),
    ("Dining Hubs", "MB Food Square (MBFS), Central Mess 2", "310 bays"),
    ("Residential", "GH5 (Girls Hostel), BH2 (Boys Hostel), Sports Ground", "440 bays")
]

for row_idx, row in enumerate(zone_data, start=1):
    for col_idx, val in enumerate(row):
        cell = table.cell(row_idx, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_WHITE if row_idx % 2 == 1 else RGBColor(241, 245, 249)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = FONT_FAMILY
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT_DARK

add_speaker_notes(s2, (
    "SPEAKER SCRIPT (0:45):\n"
    "Conventional smart parking systems in research papers rely on individual bay sensors. "
    "On our 600-acre campus, parking occurs on open gravel lots across 12 zones.\n"
    "Static systems fail due to two main phenomena: abrupt lecture bell arrival spikes where lot occupancy jumps "
    "by 300% in minutes, and the herd collapse phenomenon where naive navigation sends every driver to the same empty lot."
))

# ===========================================================================
# SLIDE 3: PERCEPTION LAYER & GEOMETRY
# ===========================================================================
s3 = prs.slides.add_slide(blank_layout)
set_slide_bg(s3, prs, C_BG_LIGHT)
add_header(s3, "Module 02 : Computer Vision & Edge Perception", "Dual YOLO Detection, WebGPU & Ground-Contact Geometry", "Overcoming camera degradation, perspective bounding-box distortion, and multi-camera overlap")

# 3 Column Layout
# Col 1: Quality Gate & WebGPU
add_card(s3, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.2))
tb = s3.shapes.add_textbox(Inches(0.95), Inches(1.95), Inches(3.3), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True
add_bullet(tf, "Pre-Inference Quality Gate", 
           "Surveillance feeds suffer from lens fog, headlight glare, and dust. OpenCV evaluates Laplacian variance:\nVar(∇² I) < τ_blur\nRejects degraded frames before ML inference.", is_first=True)
add_bullet(tf, "Dual YOLO Engines", 
           "• Fine-Tuned Model: best.pt (91.6% mAP@0.5) trained on campus terrain for dedicated car detection.\n• Pretrained yolo26l.pt: Fallback multi-class detection (buses, trucks, bikes).")
add_bullet(tf, "Client-Side WebGPU YOLO", 
           "Runs hardware-accelerated YOLO in-browser via WebGPU/WASM, enabling zero-cloud GPU hosting on Render/Vercel.")

# Col 2: Ground-Contact Point Jordan PIP
add_card(s3, Inches(4.7), Inches(1.8), Inches(4.2), Inches(5.2))
tb = s3.shapes.add_textbox(Inches(4.85), Inches(1.95), Inches(3.9), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True
add_bullet(tf, "The Perspective Distortion Trap", 
           "Surveillance cameras view parking lots at oblique angles. When a tall SUV is parked outside an ROI boundary, its roof extends into the polygon. Using Bounding Box Center produces false positive counts.", is_first=True)
add_bullet(tf, "Ground-Contact Point Projection", 
           "We anchor vehicles strictly at the tire-ground intersection:\n\nP_contact = ((x_min + x_max)/2, y_max)\n\nThis maps the exact physical footprint of the vehicle onto the parking ground.")
add_bullet(tf, "Jordan Ray-Casting PIP", 
           "A horizontal ray is cast from P_contact; intersection parity with polygon edges proves exact containment inside arbitrary non-rectangular lot boundaries.")

# Col 3: Cross-Camera Spatial NMS & Metrics Callout
add_card(s3, Inches(9.2), Inches(1.8), Inches(3.3), Inches(5.2))
# Metric Box 1: mAP
kpi1 = add_card(s3, Inches(9.4), Inches(2.0), Inches(2.9), Inches(1.4), bg_color=C_NAVY_SLATE, border_color=None)
tb = s3.shapes.add_textbox(Inches(9.5), Inches(2.1), Inches(2.7), Inches(1.2))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "91.6%"
p.font.name = FONT_FAMILY
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = C_EMERALD
p2 = tf.add_paragraph()
p2.text = "mAP@0.5 Detection Precision (best.pt on campus lots)"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(10)
p2.font.color.rgb = C_WHITE

# Metric Box 2: Spatial NMS
kpi2 = add_card(s3, Inches(9.4), Inches(3.6), Inches(2.9), Inches(1.4), bg_color=C_NAVY_SLATE, border_color=None)
tb = s3.shapes.add_textbox(Inches(9.5), Inches(3.7), Inches(2.7), Inches(1.2))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "d < 2.5 m"
p.font.name = FONT_FAMILY
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = C_ACCENT_CYAN
p2 = tf.add_paragraph()
p2.text = "Cross-Camera Global Spatial NMS Deduplication Radius"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(10)
p2.font.color.rgb = C_WHITE

# Detail below boxes
tb_d = s3.shapes.add_textbox(Inches(9.4), Inches(5.2), Inches(2.9), Inches(1.6))
tf_d = tb_d.text_frame
tf_d.word_wrap = True
p = tf_d.paragraphs[0]
p.text = "Multi-Camera Fusion:"
p.font.name = FONT_FAMILY
p.font.size = Pt(11)
p.font.bold = True
p.font.color.rgb = C_NAVY_DARK
p2 = tf_d.add_paragraph()
p2.text = "When adjacent cameras monitor overlapping ground sections, Global NMS suppresses duplicate detections within 2.5m Euclidean ground distance."
p2.font.name = FONT_FAMILY
p2.font.size = Pt(10)
p2.font.color.rgb = C_TEXT_MUTED

add_speaker_notes(s3, (
    "SPEAKER SCRIPT (1:00):\n"
    "Our vision layer begins with an OpenCV quality filter that checks blur and glare before inference. "
    "We trained a custom YOLO model achieving 91.6% mAP50, and also implemented in-browser WebGPU YOLO "
    "so client machines run inference without server GPU costs.\n"
    "Crucially, bounding box centers fail on angled surveillance cameras because vehicle roofs cross ROI boundaries. "
    "We evaluate containment strictly at the bottom-center tire contact point using Jordan ray casting, "
    "and suppress multi-camera duplicates within a 2.5-meter radius."
))

# ===========================================================================
# SLIDE 4: ZERO-LEAKAGE FEATURE PIPELINE
# ===========================================================================
s4 = prs.slides.add_slide(blank_layout)
set_slide_bg(s4, prs, C_BG_LIGHT)
add_header(s4, "Module 03 : Time-Series ML Engineering", "Zero-Leakage Feature Pipeline (41 Causal Features)", "Unified scikit-learn ColumnTransformer eliminating lookahead leakage and cross-zone lag bleeding")

# Card 1: Feature Taxonomy (Left)
add_card(s4, Inches(0.8), Inches(1.8), Inches(6.8), Inches(5.2))
tb = s4.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(6.4), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True

add_bullet(tf, "1. Harmonic Cyclics (No Midnight Discontinuity)",
           "hour_sin = sin(2π · hour / 24),  hour_cos = cos(2π · hour / 24)\nday_sin = sin(2π · dow / 7),    day_cos = cos(2π · dow / 7)\nEncodes diurnal and weekly rhythms smoothly without boundary cliffs.", is_first=True)
add_bullet(tf, "2. Causal Auto-Regressive Lags & Velocity",
           "• Strict historical lags: y_{t-15}, y_{t-30}, y_{t-45}, y_{t-60}\n• Demand Velocity: Δ_{15m} = y_t - y_{t-15} and Δ_{30m} = y_t - y_{t-30}\nCaptures immediate inflow momentum rather than static occupancy alone.")
add_bullet(tf, "3. Shifted Rolling Windows (Zero Leakage)",
           "Rolling means and standard deviations (30m, 60m) are computed strictly over y_{t-1} backward, ensuring no current-step leakage.")
add_bullet(tf, "4. Leave-One-Week-Out Normal-Day Profile",
           "Baseline historical profile per zone and time-of-week, leaving out the target week during training to establish an uncorrupted expected baseline.")

# Card 2: Lookahead Calendar & Architecture (Right)
add_card(s4, Inches(7.9), Inches(1.8), Inches(4.6), Inches(5.2))
tb_r = s4.shapes.add_textbox(Inches(8.1), Inches(1.95), Inches(4.2), Inches(4.9))
tf_r = tb_r.text_frame
tf_r.word_wrap = True

add_bullet(tf_r, "Arrival-Time Known Lookaheads", 
           "Because the campus calendar and class timetables are scheduled in advance, the model receives deterministic future knowledge at arrival time (t + horizon):\n• Distance to next lecture bell\n• Scheduled exam day flag\n• Campus event load / holiday", is_first=True)
add_bullet(tf_r, "Groupby('zone_id') Bleed Guard", 
           "When training data is multi-zone, shifting across raw rows causes lag bleeding between lots. Grouping by zone_id guarantees complete topological isolation.")
add_bullet(tf_r, "Unified Persisted ColumnTransformer", 
           "All one-hot zone encodings, standard scaling, and cyclical transforms are bundled into a single persisted sklearn Pipeline object, preventing any train-serve skew.")

add_speaker_notes(s4, (
    "SPEAKER SCRIPT (1:00):\n"
    "Data leakage is the single most common pitfall in time-series forecasting. "
    "We engineered 41 causal features: continuous harmonic sines and cosines, causal lags up to 60 minutes, "
    "and rolling statistics calculated strictly over t-1 backward.\n"
    "We also incorporate a leave-one-week-out normal-day profile and exploit deterministic arrival-time "
    "knowledge like class bells and campus events. All feature engineering executes inside a zone-grouped "
    "scikit-learn ColumnTransformer, guaranteeing zero train-serve skew."
))

# ===========================================================================
# SLIDE 5: VALIDATION & OPTUNA TUNING
# ===========================================================================
s5 = prs.slides.add_slide(blank_layout)
set_slide_bg(s5, prs, C_BG_LIGHT)
add_header(s5, "Module 04 : Methodology & Validation", "Chronological Splitting & Optuna Bayesian Tuning", "Enforcing 1-hour temporal purge buffers and TimeSeriesSplit expanding window cross-validation")

# Card 1: Chronological Split (Left)
add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
tb = s5.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True

add_bullet(tf, "The Flaw of Random K-Fold Splits", 
           "Shuffling time-series observations randomly allows future data points to train models predicting past events, inflating validation scores artificially.", is_first=True)
add_bullet(tf, "70% / 15% / 15% Chronological Partitioning", 
           "Data is partitioned strictly by calendar time:\n• Train Set: First 70% of chronological observations\n• Validation Set: Middle 15% of timeline\n• Holdout Test Set: Final untouched 15% of timeline")
add_bullet(tf, "The 1-Hour Temporal Purge Buffer", 
           "To prevent maximum lag (60 min) from bridging split boundaries, a full 1-hour temporal window is purged between Train-Val and Val-Test splits. No feature ever touches data across the gap.")

# Card 2: Optuna Bayesian Hyperparameter Tuning (Right)
add_card(s5, Inches(6.7), Inches(1.8), Inches(5.8), Inches(5.2))
tb_t = s5.shapes.add_textbox(Inches(6.9), Inches(1.95), Inches(5.4), Inches(4.9))
tf_t = tb_t.text_frame
tf_t.word_wrap = True

add_bullet(tf_t, "5-Fold Expanding TimeSeriesSplit", 
           "Hyperparameters were cross-validated using TimeSeriesSplit(n_splits=5, gap=12 steps). The 12-step gap (1 hour) guarantees zero autocorrelation bleed between expanding folds.", is_first=True)
add_bullet(tf_t, "Optuna TPE Optimization Results", 
           "Bayesian Tree-structured Parzen Estimator (TPE) optimized multi-horizon XGBoost hyperparameters:\n• max_depth = 5 (restricts over-complex leaf splits)\n• learning_rate = 0.05 (stable gradient convergence)\n• n_estimators = 200 (sufficient boosting depth)\n• subsample = 0.85, colsample_bytree = 0.85")
add_bullet(tf_t, "Centralized Config Overrides", 
           "Tuned parameters are persisted to configs/tuned_params.yaml, seamlessly overriding baseline defaults in model_config.yaml during runtime.")

add_speaker_notes(s5, (
    "SPEAKER SCRIPT (0:45):\n"
    "Random k-fold validation creates massive lookahead bias. We partitioned our 18-week continuous dataset "
    "chronologically into 70% Train, 15% Validation, and 15% Holdout Test.\n"
    "Crucially, we dropped a 1-hour temporal purge buffer between splits so no 60-minute lag could cross the boundary. "
    "For hyperparameter tuning, we used Optuna's Bayesian TPE algorithm across an expanding 5-fold TimeSeriesSplit "
    "with a 12-step gap, converging on depth 5 and 200 estimators."
))

# ===========================================================================
# SLIDE 6: BENCHMARKS & MULTI-HORIZON RESULTS
# ===========================================================================
s6 = prs.slides.add_slide(blank_layout)
set_slide_bg(s6, prs, C_BG_LIGHT)
add_header(s6, "Module 05 : Empirical Results", "Multi-Model Benchmark & Forecast Accuracy", "Evaluating candidate models across +15, +30, and +60 min horizons on the untouched Holdout Test Split")

# Full Table Card
add_card(s6, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.6))
tb_tbl_title = s6.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(11.3), Inches(0.35))
p = tb_tbl_title.text_frame.paragraphs[0]
p.text = "Holdout Test Split Benchmark Table (data/processed/lpu_timeseries.csv)"
p.font.name = FONT_FAMILY
p.font.size = Pt(12)
p.font.bold = True
p.font.color.rgb = C_PRIMARY_BLUE

# Benchmark Table (8 rows, 6 columns)
t_shape = s6.shapes.add_table(8, 6, Inches(1.0), Inches(2.3), Inches(11.3), Inches(2.9))
t = t_shape.table
t.columns[0].width = Inches(2.8)
t.columns[1].width = Inches(1.6)
t.columns[2].width = Inches(1.7)
t.columns[3].width = Inches(1.7)
t.columns[4].width = Inches(1.7)
t.columns[5].width = Inches(1.8)

bench_headers = ["Model Architecture", "Horizon", "MAE (Cars)", "RMSE (Cars)", "R² Score", "Skill vs Persistence"]
for c_idx, h in enumerate(bench_headers):
    cell = t.cell(0, c_idx)
    cell.fill.solid()
    cell.fill.fore_color.rgb = C_NAVY_SLATE
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

bench_rows = [
    ("Seasonal Mean (Historical)", "+15 min", "15.009", "31.573", "0.8007", "-170.2%"),
    ("Persistence Baseline (y_t)", "+15 / +30 / +60m", "5.56 / 8.91 / 14.68", "12.4 / 19.0 / 29.4", "0.969 / 0.928 / 0.828", "Baseline (0%)"),
    ("Normal-Day Profile", "+15 / +60 min", "8.326 / 8.325", "17.426", "0.9393", "-49.9% / +43.3%"),
    ("Ridge Regression (L2)", "+15 / +30 / +60m", "4.483 / 6.088 / 8.216", "10.09 / 13.34 / 17.09", "0.980 / 0.964 / 0.942", "+19% / +32% / +44%"),
    ("Random Forest (Bagging)", "+15 / +30 / +60m", "3.804 / 4.614 / 5.553", "6.09 / 7.09 / 8.44", "0.993 / 0.990 / 0.986", "+32% / +48% / +62%"),
    ("Tuned XGBoost (Validation)", "+15 min", "3.542", "5.996", "0.9930", "+36.2%"),
    ("Tuned XGBoost (Holdout Test)", "+15 / +30 / +60m", "3.485 / 4.151 / 4.776", "5.19 / 6.18 / 6.93", "0.995 / 0.992 / 0.990", "+37.3% / +53.4% / +67.5%")
]

for r_idx, r in enumerate(bench_rows, start=1):
    is_winner = (r_idx == 7)
    for c_idx, val in enumerate(r):
        cell = t.cell(r_idx, c_idx)
        cell.fill.solid()
        if is_winner:
            cell.fill.fore_color.rgb = RGBColor(220, 252, 231)  # Soft emerald highlight
        else:
            cell.fill.fore_color.rgb = C_WHITE if r_idx % 2 == 1 else RGBColor(248, 250, 252)
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = FONT_FAMILY
        p.font.size = Pt(9.5)
        p.font.bold = is_winner
        p.font.color.rgb = C_NAVY_DARK if not is_winner else RGBColor(6, 95, 70)

# Bottom Row: Ablation Study Findings Card
add_card(s6, Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.4), bg_color=C_NAVY_SLATE, border_color=None)
tb_ab = s6.shapes.add_textbox(Inches(1.0), Inches(5.7), Inches(11.3), Inches(1.2))
tf_ab = tb_ab.text_frame
p = tf_ab.paragraphs[0]
p.text = "KEY TAKEAWAY: FORECAST ABLATION STUDY (experiments/forecast_ablation.py)"
p.font.name = FONT_FAMILY
p.font.size = Pt(10)
p.font.bold = True
p.font.color.rgb = C_EMERALD

p2 = tf_ab.add_paragraph()
p2.text = "• Boosting decisively outperforms Bagging (Random Forest) at all horizons (MAE 4.78 vs 5.55 at +60m)."
p2.font.name = FONT_FAMILY
p2.font.size = Pt(11)
p2.font.color.rgb = C_WHITE

p3 = tf_ab.add_paragraph()
p3.text = "• Ensembling / Stacking Random Forest with XGBoost yielded 0% gain (blend weights collapsed 77-91% onto XGBoost)."
p3.font.name = FONT_FAMILY
p3.font.size = Pt(11)
p3.font.color.rgb = RGBColor(203, 213, 225)

add_speaker_notes(s6, (
    "SPEAKER SCRIPT (1:00):\n"
    "On the unseen holdout test split, Tuned XGBoost outperforms all candidate baselines at every horizon. "
    "At +15 minutes, MAE drops to 3.48 cars, at +30 minutes to 4.15 cars, and at +60 minutes to 4.78 cars.\n"
    "Notice the Skill Score: our model improves over Persistence by 37% at 15m, 53% at 30m, and 67% at one hour. "
    "Our ablation experiments proved that Boosting decisively beats Random Forest bagging, and an ensemble stack "
    "added nothing as blend weights collapsed 91% onto XGBoost."
))

# ===========================================================================
# SLIDE 7: BELL SURGES & SATURATION HEAD
# ===========================================================================
s7 = prs.slides.add_slide(blank_layout)
set_slide_bg(s7, prs, C_BG_LIGHT)
add_header(s7, "Module 06 : Critical Regimes & Dual-Head", "Bell Surge Inflections & Isotonic Saturation Classifier", "Unmasking the persistence paradox during lecture transitions and classifying capacity saturation")

# Left: Bell Surge Inflection Analysis
add_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
tb = s7.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True

add_bullet(tf, "The 'Persistence Paradox' Exposed", 
           "During steady calm hours (e.g., midnight or mid-lecture), occupancy barely changes. Naive persistence achieves an apparently respectable MAE of 5.56 cars.\nHowever, parking models are only tested when traffic surges.", is_first=True)
add_bullet(tf, "Rush-Hour Class Transition Performance", 
           "Evaluating error strictly during ±15-minute lecture bell windows:\n• Persistence Error explodes to: 11.32 cars\n• Ridge Regression Error: 8.29 cars\n• Tuned XGBoost Error: 4.44 cars (~61% Error Reduction!)")
add_bullet(tf, "Performance Across Operational Regimes", 
           "At +60 min by day type: normal weekday 5.3 cars, weekend 4.0, holiday 4.2, exam day 4.6, campus event 8.9 (vs 20.5 for persistence).")

# Right: Saturation Classifier (Dual-Head Architecture)
add_card(s7, Inches(6.7), Inches(1.8), Inches(5.8), Inches(5.2))

# Metric Card 1: PR-AUC
kpi_pr = add_card(s7, Inches(6.9), Inches(2.0), Inches(2.6), Inches(1.3), bg_color=C_NAVY_SLATE, border_color=None)
tb = s7.shapes.add_textbox(Inches(7.0), Inches(2.05), Inches(2.4), Inches(1.1))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "0.9725"
p.font.name = FONT_FAMILY
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = C_EMERALD
p2 = tf.add_paragraph()
p2.text = "PR-AUC (Precision-Recall Area)"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(9.5)
p2.font.color.rgb = C_WHITE

# Metric Card 2: Recall
kpi_rec = add_card(s7, Inches(9.7), Inches(2.0), Inches(2.6), Inches(1.3), bg_color=C_NAVY_SLATE, border_color=None)
tb = s7.shapes.add_textbox(Inches(9.8), Inches(2.05), Inches(2.4), Inches(1.1))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "90.26%"
p.font.name = FONT_FAMILY
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = C_ACCENT_CYAN
p2 = tf.add_paragraph()
p2.text = "Recall at 91% Precision (Acc: 97.8%)"
p2.font.name = FONT_FAMILY
p2.font.size = Pt(9.5)
p2.font.color.rgb = C_WHITE

tb_c = s7.shapes.add_textbox(Inches(6.9), Inches(3.5), Inches(5.4), Inches(3.3))
tf_c = tb_c.text_frame
tf_c.word_wrap = True

add_bullet(tf_c, "Why Continuous Regression Alone Fails Drivers", 
           "A regression prediction of 215/240 cars carries uncertainty. That 5-car margin decides whether a driver finds parking or gets trapped in an overflowed lot.", is_first=True)
add_bullet(tf_c, "Isotonic-Calibrated Saturation Head", 
           "We trained a dedicated binary classifier predicting P(Occupancy ≥ 90%). Because saturation is rare (<6.5% of timestamps), standard ROC-AUC is misleading. We calibrated probabilities on validation data using Isotonic Regression, yielding PR-AUC of 0.9725.")

add_speaker_notes(s7, (
    "SPEAKER SCRIPT (1:00):\n"
    "Standard average metrics conceal where naive baselines break down: during lecture bell changes. "
    "Here, persistence error spikes to 11.3 cars, while our tuned XGBoost keeps error down to 4.4 cars—a 61% reduction.\n"
    "Furthermore, continuous count predictions are not enough to protect drivers from overflows. "
    "We built a dual-head saturation classifier with isotonic probability calibration, achieving a PR-AUC of 0.9725 "
    "and 90.3% recall under extreme class imbalance."
))

# ===========================================================================
# SLIDE 8: UNCERTAINTY & FAR HORIZON
# ===========================================================================
s8 = prs.slides.add_slide(blank_layout)
set_slide_bg(s8, prs, C_BG_LIGHT)
add_header(s8, "Module 07 : Decision Intelligence", "Calibrated Uncertainty & Far-Horizon Planning", "Empirical 80% confidence bands, P(free slot on arrival), and 7-day campus calendar forecasting")

# Card 1: Uncertainty & P(Free Slot)
add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2))
tb = s8.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.2), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True

add_bullet(tf, "Beyond Point Estimates", 
           "Real drivers need risk bounds, not just a single count number. We compute the empirical residual distribution (actual - forecast) conditioned on lot fullness and operational regime.", is_first=True)
add_bullet(tf, "Empirical 80% Prediction Ranges", 
           "Validated on the held-out test split, our 80% ranges achieve exact empirical coverage:\n• +15 min: 79.8% coverage\n• +30 min: 79.3% coverage\n• +60 min: 80.1% coverage\n• During events: 81.8% to 89.3% coverage")
add_bullet(tf, "P(Free Slot on Arrival)", 
           "Computes probability that at least max(2, 3% capacity) bays remain vacant on arrival. Achieves Brier scores of 0.012–0.016 against 0.056 for guessing the base rate.")

# Card 2: Far-Horizon Calendar Forecasting
add_card(s8, Inches(6.7), Inches(1.8), Inches(5.8), Inches(5.2))
tb_f = s8.shapes.add_textbox(Inches(6.9), Inches(1.95), Inches(5.4), Inches(4.9))
tf_f = tb_f.text_frame
tf_f.word_wrap = True

add_bullet(tf_f, "Clock-Time Arrival Planning (Hours Ahead)", 
           "Drivers can select target clock times (9 AM, 1 PM, 5 PM) or schedule arrivals up to 7 days in advance. Live autoregressive lags fade beyond 60 minutes.", is_first=True)
add_bullet(tf_f, "Exponential Anomaly Fading (τ = 180 min)", 
           "The far-horizon forecast blends the campus calendar model with live +60m anomalies via exponential decay:\n\nŷ_{t+H} = Calendar(t+H) + [ŷ_{t+60} - Calendar(t+60)] · e^{-(H - 60) / 180}\n\nValidation testing proved this cuts 4-hour horizon error from 31.5 cars to 7.3 cars.")
add_bullet(tf_f, "Admin Campus Calendar Integration", 
           "Special events (convocations, festivals) with expected vehicle loads and exam windows update the demand forecaster dynamically via /api/calendar.")

add_speaker_notes(s8, (
    "SPEAKER SCRIPT (1:00):\n"
    "Drivers don't just want a point estimate; they need to know if a bay will be free. "
    "We provide calibrated 80% prediction intervals that hit near-perfect 80% test coverage, and a "
    "P(free slot) probability score with a Brier score of 0.014.\n"
    "For drivers planning hours or days ahead, our far-horizon model blends the campus calendar model "
    "with live 60-minute anomalies using exponential decay (tau = 180 min), reducing 4-hour forecast error "
    "from 31.5 to 7.3 cars."
))

# ===========================================================================
# SLIDE 9: XAI & ANTI-HERDING DISPATCH
# ===========================================================================
s9 = prs.slides.add_slide(blank_layout)
set_slide_bg(s9, prs, C_BG_LIGHT)
add_header(s9, "Module 08 : Explainability & Dispatch", "Explainable AI (TreeSHAP) & Anti-Herding Dispatch", "Global feature attributions, local decision traces, and Softmax stochastic traffic dispersion")

# Left: SHAP Visual / Explanation Card
add_card(s9, Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.2))
# Check if shap beeswarm image exists
shap_img_path = Path("artifacts/figures/shap_summary_beeswarm.png")
if shap_img_path.exists():
    s9.shapes.add_picture(str(shap_img_path), Inches(1.0), Inches(2.0), Inches(5.4), Inches(2.8))
    tb_sh = s9.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(5.4), Inches(2.0))
    tf_sh = tb_sh.text_frame
    tf_sh.word_wrap = True
    p = tf_sh.paragraphs[0]
    p.text = "TreeSHAP Interpretability Findings:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY_BLUE
    p2 = tf_sh.add_paragraph()
    p2.text = "• lag_15m and normal-day profile dominate baseline anchoring.\n• delta_15m (velocity) provides non-linear upward pushes during arrival surges.\n• Live endpoint GET /api/explain outputs vehicle-unit impact per feature."
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(10)
    p2.font.color.rgb = C_TEXT_MUTED
else:
    tb_sh = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.4), Inches(4.8))
    tf_sh = tb_sh.text_frame
    add_bullet(tf_sh, "TreeSHAP Global & Local Attribution", 
               "• lag_15m and normal-day profile dominate baseline anchoring.\n• delta_15m (velocity) drives momentum pushes during sudden rushes.\n• Live endpoint GET /api/explain explains every prediction in exact vehicle units.", is_first=True)

# Right: Anti-Herding Dispatch Logic
add_card(s9, Inches(6.9), Inches(1.8), Inches(5.6), Inches(5.2))
tb_ah = s9.shapes.add_textbox(Inches(7.1), Inches(1.95), Inches(5.2), Inches(4.9))
tf_ah = tb_ah.text_frame
tf_ah.word_wrap = True

add_bullet(tf_ah, "Multi-Criteria Ranking Utility", 
           "Score(z) = 0.50 · (1 - OccRate_z) + 0.35 · (1 - d_{z, dest}/1200m) - 0.15 · Pen_camera\nBalances predicted vacancy, walking distance to destination, and camera freshness.", is_first=True)
add_bullet(tf_ah, "In-Flight Dispatch Tracker (15-min TTL)", 
           "Active driver selections (POST /api/navigate) temporarily decrement visible capacity:\nEffective Available = Capacity_z - ŷ_{z, t+H} - N_{in_flight}\nCapped at 35% of lot capacity to avoid phantom gridlock.")
add_bullet(tf_ah, "Softmax Stochastic Dispersion", 
           "When top-ranking lots are near-tied, a Softmax selector stochastically distributes traffic among the top-3 Pareto lots instead of dumping 100% of vehicles into one ground.\nReplay result: 99.8% free slot rate vs 91.9% naive nearest.")

add_speaker_notes(s9, (
    "SPEAKER SCRIPT (0:45):\n"
    "For transparency, our microservice computes TreeSHAP explanations globally and locally via GET /api/explain, "
    "showing how 15-minute velocity and class bells push predicted counts in vehicle units.\n"
    "To solve the herd collapse trap where everyone rushes to the same lot, our recommendation engine uses "
    "multi-criteria utility scoring, Softmax dispersion, and an In-Flight Dispatch Tracker with a 15-minute TTL buffer. "
    "In replay testing, our recommender achieved a 99.8% slot success rate compared to 91.9% for naive nearest routing."
))

# ===========================================================================
# SLIDE 10: PRODUCTION MICROSERVICE & ROADMAP
# ===========================================================================
s10 = prs.slides.add_slide(blank_layout)
set_slide_bg(s10, prs, C_BG_LIGHT)
add_header(s10, "Module 09 : System Architecture & Conclusion", "Production Microservice, Verification & Future Scope", "Role-based FastAPI architecture, SQLite real-world ingestion, and 26 automated unit tests")

# Left Column: Production Full-Stack Architecture
add_card(s10, Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.2))
tb = s10.shapes.add_textbox(Inches(1.0), Inches(1.95), Inches(5.4), Inches(4.9))
tf = tb.text_frame
tf.word_wrap = True

add_bullet(tf, "FastAPI Production Microservice", 
           "• Asynchronous backend with SlowAPI rate limiting & Pydantic v2 validation.\n• 5 Dedicated TailwindCSS/Chart.js frontends: Driver Assistant (/), Admin Command Center (/admin), Campus Calendar (/calendar), Vision Studio (/vision), Model Health & Drift (/model).", is_first=True)
add_bullet(tf, "Role-Based Access Control (RBAC)", 
           "Authentication at /login partitions access across Driver, Admin, and Security roles, with one-click demo sign-in for seamless evaluation.")
add_bullet(tf, "SQLite Real-Data Ingestion Pipeline", 
           "Includes POST /api/ingest and schema exporter so real camera observations seamlessly accumulate to replace synthetic data without downtime.")

# Right Column: Verification & Future Roadmap
add_card(s10, Inches(6.9), Inches(1.8), Inches(5.6), Inches(5.2))
tb_r = s10.shapes.add_textbox(Inches(7.1), Inches(1.95), Inches(5.2), Inches(4.9))
tf_r = tb_r.text_frame
tf_r.word_wrap = True

add_bullet(tf_r, "Comprehensive Automated Test Suite", 
           "Verified by 26 automated unit and integration tests (pytest tests/ -v) covering API endpoints, CQR bands, anti-herding dispatch, and computer vision filters.", is_first=True)
add_bullet(tf_r, "Practical Edge-Case Resilience", 
           "• Fallback to normal-day profiles during camera network outages.\n• Stream-freeze detector alerts operators when CCTV frames stagnate.\n• Dynamic capacity capping prevents buffer overflow.", is_first=False)
add_bullet(tf_r, "Future Engineering Roadmap", 
           "• Edge deployment of YOLO models on lamp-post mounted NVIDIA Jetson / Raspberry Pi 5 nodes.\n• Integration with university ERP and Automatic License Plate Recognition (ALPR) at campus barriers.")

add_speaker_notes(s10, (
    "SPEAKER SCRIPT (1:00):\n"
    "We delivered this entire system as an asynchronous FastAPI microservice with role-based authentication, "
    "interactive TailwindCSS dashboards, and an automated SQLite ingestion pipeline to transition from simulation to real cameras.\n"
    "The codebase is verified by 26 automated unit and integration tests. In the future, we plan to deploy our YOLO vision models "
    "directly onto edge NVIDIA Jetson nodes mounted on campus lamp posts. Thank you to Dr. Mohit Arora and the committee. "
    "We are now ready for your questions."
))

# ---------------------------------------------------------------------------
# SAVE PRESENTATION
# ---------------------------------------------------------------------------
output_dir = Path("artifacts")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "Campus_Parking_Project32_Presentation.pptx"

prs.save(str(output_path))
print(f"[SUCCESS] Presentation generated successfully: {output_path.resolve()}")
