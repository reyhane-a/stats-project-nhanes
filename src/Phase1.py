import pandas as pd
import numpy as np

DATA_DIR = "data/raw"
FILES = ["DEMO_L", "DPQ_L", "SMQ_L", "ALQ_L", "BPQ_L", "HSQ_L"]

# ===== خواندن فایل‌ها =====
raw = {}
for f in FILES:
    df = pd.read_sas(f"{DATA_DIR}/{f}.xpt", format="xport")
    # مقدارهای خیلی کوچک (مثل 5.4e-79) در واقع صفر هستند
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-10, 0)
    raw[f] = df

summary = pd.DataFrame({
    "file": FILES,
    "rows": [len(raw[f]) for f in FILES],
    "unique_SEQN": [raw[f]["SEQN"].nunique() for f in FILES],
    "columns": [raw[f].shape[1] for f in FILES],
})
print(summary)

# ===== ادغام با SEQN =====
m = raw["DEMO_L"].copy()
n_before = len(m)
for f in FILES[1:]:
    m = m.merge(raw[f], on="SEQN", how="left", validate="1:1")
print("before merge:", n_before, "| after merge:", len(m), "| SEQN unique:", m["SEQN"].is_unique)

# ===== ساخت متغیرها =====
# نمره PHQ-9: فقط اگر هر ۹ آیتم معتبر باشد. کد ۷ و ۹ گمشده حساب می‌شوند.
phq_items = [f"DPQ0{i}0" for i in range(1, 10)]
items = m[phq_items].replace({7: np.nan, 9: np.nan})
m["phq9"] = items.sum(axis=1, min_count=9)
m["phq9_ge10"] = np.where(m["phq9"].isna(), np.nan, (m["phq9"] >= 10).astype(float))

# وضعیت سیگار
s20 = m["SMQ020"].replace({7: np.nan, 9: np.nan})

# ===== ساخت متغیرها =====
# نمره PHQ-9: فقط اگر هر ۹ آیتم معتبر باشد. کد ۷ و ۹ گمشده حساب می‌شوند.
phq_items = [f"DPQ0{i}0" for i in range(1, 10)]
items = m[phq_items].replace({7: np.nan, 9: np.nan})
m["phq9"] = items.sum(axis=1, min_count=9)
m["phq9_ge10"] = np.where(m["phq9"].isna(), np.nan, (m["phq9"] >= 10).astype(float))

# وضعیت سیگار
s20 = m["SMQ020"].replace({7: np.nan, 9: np.nan})
s40 = m["SMQ040"].replace({7: np.nan, 9: np.nan})
m["smoke"] = np.select(
    [s20 == 2, (s20 == 1) & (s40 == 3), (s20 == 1) & s40.isin([1, 2])],
    ["never", "former", "current"],
    default="missing",
)
m["smoke"] = m["smoke"].replace("missing", np.nan)

m["sex"] = m["RIAGENDR"].map({1: "male", 2: "female"})

# ===== جمعیت تحلیلی و جدول جریان =====
adults = m[m["RIDAGEYR"] >= 20]
has_dpq = adults[adults["SEQN"].isin(raw["DPQ_L"]["SEQN"])]
phq_ok = has_dpq[has_dpq["phq9"].notna()]
analytic = phq_ok[phq_ok["smoke"].notna()].copy()

flow = pd.DataFrame([
    ("All DEMO_L", len(m)),
    ("Age >= 20", len(adults)),
    ("Has DPQ record", len(has_dpq)),
    ("PHQ-9 complete", len(phq_ok)),
    ("Smoking status known (analytic sample)", len(analytic)),
], columns=["step", "remaining"])
flow["removed"] = (-flow["remaining"].diff()).fillna(0).astype(int)
print(flow)

print("INDFMPIR missing:", analytic["INDFMPIR"].isna().sum(),
      f"({analytic['INDFMPIR'].isna().mean():.1%})")

# ===== نگاه اولیه =====
print(analytic["smoke"].value_counts())
print(analytic["phq9"].describe())
print(analytic.groupby("smoke")["phq9"].agg(["count", "mean", "median", "std"]).round(2))

# ======================================================
# مرحله ۲: ممیزی کیفیت داده
# ======================================================
print("\n===== AUDIT: PHQ-9 items (adults with DPQ record) =====")
aud = []
for c in phq_items:
    x = has_dpq[c]
    aud.append({"item": c, "valid(0-3)": x.isin([0, 1, 2, 3]).sum(),
                "refused(7)": (x == 7).sum(), "dont_know(9)": (x == 9).sum(),
                "missing(NaN)": x.isna().sum()})
print(pd.DataFrame(aud).to_string(index=False))

print("\n===== AUDIT: smoking variables (all adults 20+) =====")
for c in ["SMQ020", "SMQ040"]:
    x = adults[c]
    print(c, "| valid:", x.isin([1, 2, 3]).sum(), "| refused(7):", (x == 7).sum(),
          "| dont_know(9):", (x == 9).sum(), "| missing:", x.isna().sum())

print("\n===== AUDIT: logical consistency of smoking =====")
a20, a40 = adults["SMQ020"], adults["SMQ040"]
print("SMQ020=2 (never) but SMQ040 answered:", ((a20 == 2) & a40.notna()).sum())
print("SMQ020=1 (>=100 cigs) but SMQ040 missing:", ((a20 == 1) & a40.isna()).sum())

print("\n===== AUDIT: ranges (analytic sample) =====")
print("PHQ-9 min/max:", analytic["phq9"].min(), analytic["phq9"].max(), "(allowed 0-27)")
print("Age min/max:", analytic["RIDAGEYR"].min(), analytic["RIDAGEYR"].max(), "(80 = top-coded '80 and over')")
print("INDFMPIR min/max:", analytic["INDFMPIR"].min(), analytic["INDFMPIR"].max(), "(allowed 0-5, 5 = '5 or more')")
print("Weight WTMEC2YR missing or <=0:", (analytic["WTMEC2YR"].isna() | (analytic["WTMEC2YR"] <= 0)).sum())

print("\n===== AUDIT: missing % in analytic sample =====")
cols = ["phq9", "smoke", "RIDAGEYR", "sex", "INDFMPIR", "RIDRETH3", "DMDEDUC2"]
print((analytic[cols].isna().mean() * 100).round(1).to_string())

# ======================================================
# مرحله ۳: جدول فراوانی و آمار توصیفی
# ======================================================
race_map = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
            4: "NH Black", 6: "NH Asian", 7: "Other/multiracial"}
edu_map = {1: "<9th grade", 2: "9-11th grade", 3: "HS/GED", 4: "Some college/AA", 5: "College grad+"}
analytic["race"] = analytic["RIDRETH3"].map(race_map)
analytic["edu"] = analytic["DMDEDUC2"].map(edu_map)   # کدهای 7 و 9 خودکار NaN می‌شوند

def freq(col):
    n = analytic[col].value_counts(dropna=False)
    p = (n / len(analytic) * 100).round(1)
    return pd.DataFrame({"n": n, "percent": p})

for col in ["smoke", "sex", "race", "edu"]:
    print(f"\n===== FREQUENCY: {col} =====")
    print(freq(col))

print("\n===== QUANTITATIVE SUMMARY =====")
rows = []
for col in ["phq9", "RIDAGEYR", "INDFMPIR"]:
    x = analytic[col].dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    rows.append({"variable": col, "n": len(x), "mean": x.mean(), "sd": x.std(),
                 "min": x.min(), "Q1": q1, "median": x.median(), "Q3": q3,
                 "max": x.max(), "IQR": q3 - q1, "skew": x.skew()})
print(pd.DataFrame(rows).round(2).to_string(index=False))

# ======================================================
# مرحله ۴: نمودارها
# ======================================================
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
order = ["never", "former", "current"]

# نمودار ۱: هیستوگرام PHQ-9
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(analytic["phq9"], bins=range(0, 29), color="steelblue", edgecolor="white", align="left")
ax.set_xlabel("PHQ-9 total score (0-27, higher = more depressive symptoms)")
ax.set_ylabel("Number of adults")
ax.set_title("Distribution of PHQ-9 score, US adults 20+, NHANES 2021-2023 (unweighted)")
fig.tight_layout()
fig.savefig("figures/fig1_phq9_hist.png", dpi=200)
plt.close(fig)

# نمودار ۲: میله‌ای وضعیت سیگار
counts = analytic["smoke"].value_counts().reindex(order)
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(order, counts.values, color=["#4c72b0", "#dd8452", "#c44e52"])
for b, n in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, n + 30, f"{n} ({n / counts.sum() * 100:.1f}%)", ha="center")
ax.set_xlabel("Smoking status")
ax.set_ylabel("Number of adults")
ax.set_title("Smoking status among analytic sample (n = %d)" % counts.sum())
ax.set_ylim(0, counts.max() * 1.12)
fig.tight_layout()
fig.savefig("figures/fig2_smoking_bar.png", dpi=200)
plt.close(fig)

# نمودار ۳: جعبه‌ای PHQ-9 به تفکیک وضعیت سیگار
fig, ax = plt.subplots(figsize=(6.5, 4.5))
data = [analytic.loc[analytic["smoke"] == g, "phq9"] for g in order]
ax.boxplot(data, showmeans=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(order)
ax.set_xlabel("Smoking status")
ax.set_ylabel("PHQ-9 total score (0-27)")
ax.set_title("PHQ-9 score by smoking status (unweighted; triangle = mean)")
fig.tight_layout()
fig.savefig("figures/fig3_phq9_by_smoking_box.png", dpi=200)
plt.close(fig)

# نمودار ۴: هیستوگرام سن
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(analytic["RIDAGEYR"], bins=range(20, 85, 5), color="seagreen", edgecolor="white")
ax.set_xlabel("Age (years; 80 = '80 and over')")
ax.set_ylabel("Number of adults")
ax.set_title("Age distribution of analytic sample")
fig.tight_layout()
fig.savefig("figures/fig4_age_hist.png", dpi=200)
plt.close(fig)

print("Saved 4 figures in figures/")