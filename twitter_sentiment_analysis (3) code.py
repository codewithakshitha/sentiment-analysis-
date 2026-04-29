# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
"""
Twitter Sentiment Analysis - Self-Contained Version
Uses a built-in lexicon-based approach (no external ML libs needed)
"""

import re, math, os, pandas as pd, numpy as np
import matplotlib

# Save outputs next to this script (works on Windows, Mac, Linux)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from collections import Counter

# ── Built-in Sentiment Lexicon ──────────────────────────────────────────────
POS_WORDS = {
    "love":2.5,"amazing":2.8,"wonderful":2.6,"best":2.3,"great":2.1,
    "excellent":2.7,"fantastic":2.9,"happy":2.4,"joy":2.5,"thrilled":2.8,
    "excited":2.3,"breathtaking":3.0,"bliss":2.9,"grateful":2.4,"epic":2.6,
    "unbelievable":2.0,"favourite":2.1,"cozy":1.8,"adorable":2.2,"delight":2.5,
    "awesome":2.6,"incredible":2.7,"perfect":2.8,"brilliant":2.5,"enjoy":1.9,
    "beautiful":2.3,"glad":1.8,"pleased":1.7,"terrific":2.5,"superb":2.7,
    "outstanding":2.6,"magnificent":2.8,"cheerful":2.1,"wonderful":2.6,
    "pure":1.5,"new":0.5,"dreams":1.5,"recommend":1.8,"favorite":2.1,
}
NEG_WORDS = {
    "terrible":2.8,"awful":2.7,"worst":3.0,"frustrated":2.2,"disappointed":2.4,
    "horrible":2.9,"dreadful":2.8,"nightmare":2.6,"hate":2.9,"angry":2.3,
    "disgusting":2.8,"unacceptable":2.5,"crashing":1.8,"delayed":1.5,
    "slow":1.4,"cold":1.2,"bad":1.9,"difficult":1.3,"terrible":2.8,
    "never":1.5,"grey":1.2,"headache":1.8,"exhaustion":1.2,"miss":1.3,
    "broken":1.8,"poor":1.7,"raining":0.8,"unfortunately":1.5,"wrong":1.6,
}
INTENSIFIERS = {"absolutely":1.5,"really":1.3,"so":1.2,"very":1.3,"completely":1.4,"pure":1.2}
NEGATORS = {"not","never","no","don't","doesn't","didn't","won't","can't","nothing"}
EMOJIS_POS = {"😍":2.5,"🎉":2.0,"🌟":2.0,"🏆":2.5,"❤️":2.5,"💙":2.0,"🙌":2.3,"☀️":1.5,"✨":1.8,"🚗":0.5,"🐶":2.0,"☕":1.5,"🎬":1.5,"🌊":1.5,"🏃":1.5}
EMOJIS_NEG = {}

def sentiment_score(text):
    words = text.lower().split()
    score = 0; count = 0
    i = 0
    while i < len(words):
        w = re.sub(r"[^\w']", "", words[i])
        multiplier = 1.0
        # check negator window
        prev = [re.sub(r"[^\w']","",words[j]) for j in range(max(0,i-3),i)]
        if any(p in NEGATORS for p in prev):
            multiplier = -0.8
        # intensifier
        if i > 0:
            prev_w = re.sub(r"[^\w']","",words[i-1])
            if prev_w in INTENSIFIERS:
                multiplier *= INTENSIFIERS[prev_w]
        if w in POS_WORDS:
            score += POS_WORDS[w] * multiplier; count += 1
        elif w in NEG_WORDS:
            score -= NEG_WORDS[w] * multiplier; count += 1
        i += 1
    # emojis
    for emoji, val in EMOJIS_POS.items():
        if emoji in text:
            score += val; count += 1
    for emoji, val in EMOJIS_NEG.items():
        if emoji in text:
            score -= val; count += 1
    # normalise to [-1, 1]
    if count == 0: return 0.0
    raw = score / (count * 3.0)
    return max(-1.0, min(1.0, raw))

def label(score):
    if score > 0.05: return "Positive"
    if score < -0.05: return "Negative"
    return "Neutral"

def subjectivity(text):
    words = text.lower().split()
    hits = sum(1 for w in words if re.sub(r"[^\w']","",w) in POS_WORDS or re.sub(r"[^\w']","",w) in NEG_WORDS)
    return min(1.0, hits / max(len(words), 1) * 3)

# ── Dataset ──────────────────────────────────────────────────────────────────
TWEETS = [
    ("I absolutely love the new iPhone! Best purchase I've made all year 😍","Technology"),
    ("This traffic is absolutely terrible. Worst commute ever. #frustrated","Transport"),
    ("Just had an okay lunch. Nothing special, nothing bad.","Food"),
    ("The movie was absolutely breathtaking! 10/10 would recommend 🎬✨","Entertainment"),
    ("Can't believe how awful the customer service was today. Never coming back.","Business"),
    ("The weather today is pretty average, not too hot not too cold.","Weather"),
    ("So excited for the weekend! Road trip with friends 🚗🎉","Social"),
    ("Really disappointed with the product quality. Expected much better.","Business"),
    ("The conference was informative. Learned a few new things.","Professional"),
    ("This is the happiest I've been in years! Life is wonderful 🌟","Personal"),
    ("Terrible experience at the restaurant. Food was cold, service was slow.","Food"),
    ("The new software update is fine, does what it needs to do.","Technology"),
    ("Absolutely thrilled to announce my new job! Dreams do come true! 🙌","Professional"),
    ("Flight delayed again. Airport is a nightmare. So frustrating.","Transport"),
    ("The book was decent. Some parts were interesting.","Entertainment"),
    ("Best day ever spent at the beach! Pure bliss ☀️🌊","Personal"),
    ("Why do companies make returns so difficult? Completely unacceptable.","Business"),
    ("The presentation went okay. Could have been better.","Professional"),
    ("Just adopted a puppy! So much joy in my life right now 🐶❤️","Personal"),
    ("Terrible headache all day. Nothing is working.","Health"),
    ("The game last night was epic! Unbelievable comeback! 🏆","Sports"),
    ("Not sure how I feel about the new policy changes at work.","Professional"),
    ("Grateful for amazing friends who always have my back 💙","Social"),
    ("The app keeps crashing. Worst user experience ever.","Technology"),
    ("Had a neutral day. Just went through the motions.","Personal"),
    ("This coffee shop is my new favourite spot! Cozy vibes ☕","Food"),
    ("Politicians never listen to their constituents. Disgusting.","Politics"),
    ("The seminar was average - nothing I hadn't heard before.","Professional"),
    ("Finally finished my marathon! Pure joy and exhaustion 🏃","Sports"),
    ("Raining again. My mood is as grey as the sky.","Weather"),
]

base = datetime(2024, 3, 1)
rows = []
for i, (tweet, topic) in enumerate(TWEETS):
    sc = sentiment_score(tweet)
    sub = subjectivity(tweet)
    rows.append({
        "id": i+1, "tweet": tweet, "topic": topic,
        "timestamp": base + timedelta(hours=i*3),
        "score": round(sc,4), "label": label(sc),
        "subjectivity": round(sub,3),
        "polarity": round(sc,3),
    })

df = pd.DataFrame(rows)

# ── Console Output ────────────────────────────────────────────────────────────
print("=" * 65)
print("          TWITTER SENTIMENT ANALYSIS — RESULTS")
print("=" * 65)
print(f"\n  Total tweets analysed : {len(df)}")
print(f"  Date range            : {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"  Topics covered        : {df['topic'].nunique()}")

counts = df["label"].value_counts()
total  = len(df)
print("\n  ── Sentiment Distribution ──────────────────────────")
for lbl in ["Positive","Neutral","Negative"]:
    c = counts.get(lbl,0)
    pct = c/total*100
    bar = "█"*int(pct/3)
    print(f"  {lbl:<10} {c:>3} tweets  ({pct:5.1f}%)  {bar}")

print(f"\n  Average polarity score : {df['score'].mean():+.3f}")
print(f"  Average subjectivity   : {df['subjectivity'].mean():.3f}")

print("\n  ── Top 5 Most Positive Tweets ──────────────────────")
for _, r in df.nlargest(5,"score").iterrows():
    print(f"  [{r['score']:+.3f}] {r['tweet'][:65]}...")

print("\n  ── Top 5 Most Negative Tweets ──────────────────────")
for _, r in df.nsmallest(5,"score").iterrows():
    print(f"  [{r['score']:+.3f}] {r['tweet'][:65]}...")

print("\n  ── Sentiment by Topic ──────────────────────────────")
topic_grp = df.groupby("topic")["score"].mean().sort_values(ascending=False)
for topic, mean_sc in topic_grp.items():
    emoji = "[+]" if mean_sc>0.05 else ("[-]" if mean_sc<-0.05 else "[~]")
    print(f"  {emoji} {topic:<15} {mean_sc:+.3f}")

# ── Visualisation ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#0f172a")

COLORS = {"Positive":"#22c55e","Neutral":"#94a3b8","Negative":"#ef4444"}
PANEL  = "#1e293b"
TEXT   = "#f1f5f9"

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_title(title, color=TEXT, fontweight="bold", fontsize=11, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

# ── 1. Pie ────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(2,3,1)
ax1.set_facecolor(PANEL)
vc = df["label"].value_counts()
wedge_props = {"linewidth":2,"edgecolor":"#0f172a"}
wedges, texts, autotexts = ax1.pie(
    vc, labels=vc.index, colors=[COLORS[l] for l in vc.index],
    autopct="%1.1f%%", startangle=90, wedgeprops=wedge_props,
    textprops={"color":TEXT,"fontsize":10})
for at in autotexts:
    at.set_color("#0f172a"); at.set_fontweight("bold")
ax1.set_title("Sentiment Distribution", color=TEXT, fontweight="bold", fontsize=11)

# ── 2. Bar by Topic ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(2,3,2)
styled_ax(ax2,"Avg Sentiment by Topic")
topic_avg = df.groupby("topic")["score"].mean().sort_values()
bar_colors = [COLORS["Positive"] if v>0.05 else (COLORS["Negative"] if v<-0.05 else COLORS["Neutral"]) for v in topic_avg]
bars = ax2.barh(topic_avg.index, topic_avg.values, color=bar_colors, edgecolor="#0f172a", linewidth=0.5)
ax2.axvline(0, color="#475569", linewidth=1)
ax2.set_xlabel("Avg Polarity Score", color=TEXT, fontsize=9)
for label_t in ax2.get_xticklabels(): label_t.set_color(TEXT)
for label_t in ax2.get_yticklabels(): label_t.set_color(TEXT)

# ── 3. Score Histogram ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(2,3,3)
styled_ax(ax3,"Score Distribution")
for lbl, color in COLORS.items():
    data = df[df["label"]==lbl]["score"]
    ax3.hist(data, bins=8, color=color, alpha=0.75, label=lbl, edgecolor="#0f172a")
ax3.axvline(0.05, color="#22c55e", linestyle="--", alpha=0.5, linewidth=1)
ax3.axvline(-0.05, color="#ef4444", linestyle="--", alpha=0.5, linewidth=1)
ax3.set_xlabel("Polarity Score", color=TEXT, fontsize=9)
ax3.set_ylabel("Count", color=TEXT, fontsize=9)
ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
for l in ax3.get_xticklabels()+ax3.get_yticklabels(): l.set_color(TEXT)

# ── 4. Scatter: polarity vs subjectivity ─────────────────────────────────────
ax4 = fig.add_subplot(2,3,4)
styled_ax(ax4,"Polarity vs Subjectivity")
for lbl, grp in df.groupby("label"):
    ax4.scatter(grp["polarity"], grp["subjectivity"],
                c=COLORS[lbl], label=lbl, alpha=0.85, s=90,
                edgecolors="white", linewidths=0.5)
ax4.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
ax4.axhline(0.5, color="#475569", linewidth=0.8, linestyle="--")
ax4.set_xlabel("Polarity", color=TEXT, fontsize=9)
ax4.set_ylabel("Subjectivity", color=TEXT, fontsize=9)
ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
for l in ax4.get_xticklabels()+ax4.get_yticklabels(): l.set_color(TEXT)

# ── 5. Timeline ───────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(2,3,5)
styled_ax(ax5,"Sentiment Score Over Time")
for lbl, grp in df.groupby("label"):
    ax5.scatter(grp["timestamp"], grp["score"], c=COLORS[lbl], label=lbl, s=60, alpha=0.85, edgecolors="white", linewidths=0.3)
ax5.plot(df["timestamp"], df["score"].rolling(5,center=True,min_periods=1).mean(),
         color="#f59e0b", linewidth=2, alpha=0.8, label="Trend")
ax5.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
ax5.set_xlabel("Date", color=TEXT, fontsize=9)
ax5.set_ylabel("Score", color=TEXT, fontsize=9)
ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
for l in ax5.get_xticklabels()+ax5.get_yticklabels(): l.set_color(TEXT)
plt.setp(ax5.get_xticklabels(), rotation=20, ha="right")

# ── 6. Stacked bar by topic (positive/neutral/negative) ─────────────────────
ax6 = fig.add_subplot(2,3,6)
styled_ax(ax6,"Label Mix by Topic")
topics = df["topic"].unique()
pos_c = [len(df[(df["topic"]==t)&(df["label"]=="Positive")]) for t in topics]
neu_c = [len(df[(df["topic"]==t)&(df["label"]=="Neutral")]) for t in topics]
neg_c = [len(df[(df["topic"]==t)&(df["label"]=="Negative")]) for t in topics]
y = np.arange(len(topics))
ax6.barh(y, pos_c, color="#22c55e", label="Positive", edgecolor="#0f172a")
ax6.barh(y, neu_c, left=pos_c, color="#94a3b8", label="Neutral", edgecolor="#0f172a")
ax6.barh(y, neg_c, left=[p+n for p,n in zip(pos_c,neu_c)], color="#ef4444", label="Negative", edgecolor="#0f172a")
ax6.set_yticks(y); ax6.set_yticklabels(topics, fontsize=8)
ax6.set_xlabel("Tweet Count", color=TEXT, fontsize=9)
ax6.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
for l in ax6.get_xticklabels()+ax6.get_yticklabels(): l.set_color(TEXT)

# ── Title ────────────────────────────────────────────────────────────────────
fig.suptitle("Twitter Sentiment Analysis Dashboard", color=TEXT, fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(OUT_DIR, "sentiment_dashboard.png"), dpi=150, bbox_inches="tight", facecolor="#0f172a")
print("\nDashboard saved: sentiment_dashboard.png")
plt.show()  # Opens interactive chart window

df.to_csv(os.path.join(OUT_DIR, "sentiment_results.csv"), index=False)
print("CSV saved: sentiment_results.csv")
print("\nDone!")
