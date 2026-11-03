"""Step 2 - Exploratory Data Analysis.

Answers the 20 EDA questions of the project brief plus the four Step-2 analyses
(price trends by city, area vs return, crime/safety proxy, infrastructure vs
resale value).  Every chart is saved to reports/eda and the numeric answers are
written to reports/eda_findings.md.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config as C  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PAL = "viridis"
FINDINGS: list[str] = []


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def strength(r: float) -> str:
    a = abs(r)
    if a < 0.05:
        return "negligible"
    if a < 0.20:
        return "weak"
    if a < 0.50:
        return "moderate"
    return "strong"


def save(fig, name: str) -> str:
    C.EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.EDA_DIR / f"{name}.png", dpi=100)
    plt.close(fig)
    return f"reports/eda/{name}.png"


def note(title: str, img: str, text: str) -> None:
    FINDINGS.append(f"### {title}\n\n![{title}]({img})\n\n{text}\n")


def corr_text(x: pd.Series, y: pd.Series, label: str) -> tuple[float, str]:
    r = float(x.corr(y))
    return r, f"Pearson r between {label} = **{r:+.3f}** ({strength(r)})."


def iqr_outliers(s: pd.Series) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


# ----------------------------------------------------------------------------
# 1-5  price & size
# ----------------------------------------------------------------------------
def q1_to_q5(d: pd.DataFrame, sample: pd.DataFrame) -> None:
    FINDINGS.append("## 1-5. Price & Size Analysis\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(d["Price_in_Lakhs"], bins=50, kde=True, color="#3b6ea5", ax=ax)
    ax.set(title="Q1. Distribution of property prices", xlabel="Price (lakhs)")
    s = d["Price_in_Lakhs"]
    img = save(fig, "q01_price_distribution")
    note("Q1. What is the distribution of property prices?", img,
         f"Prices span {s.min():.1f} to {s.max():.1f} lakhs (mean {s.mean():.1f}, median {s.median():.1f}, "
         f"skew {s.skew():+.2f}). The distribution is almost flat, i.e. close to uniform, rather than the "
         "right-skewed shape typical of real markets.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(d["Size_in_SqFt"], bins=50, kde=True, color="#2a9d8f", ax=ax)
    ax.set(title="Q2. Distribution of property sizes", xlabel="Size (sq ft)")
    s = d["Size_in_SqFt"]
    img = save(fig, "q02_size_distribution")
    note("Q2. What is the distribution of property sizes?", img,
         f"Sizes range from {s.min():,} to {s.max():,} sq ft (mean {s.mean():,.0f}, median {s.median():,.0f}, "
         f"skew {s.skew():+.2f}); again close to uniform.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=d, x="Property_Type", y="Price_per_SqFt_INR", hue="Property_Type",
                palette=PAL, legend=False, showfliers=False, ax=ax)
    ax.set(title="Q3. Price per sq ft by property type (capped, INR)", ylabel="INR per sq ft")
    t = d.groupby("Property_Type")["Price_per_SqFt_INR"].agg(["mean", "median"]).round(0)
    img = save(fig, "q03_ppsf_by_type")
    rows = "\n".join(f"| {i} | {r['mean']:,.0f} | {r['median']:,.0f} |" for i, r in t.iterrows())
    note("Q3. How does price per sq ft vary by property type?", img,
         "| Property type | Mean INR/sqft | Median INR/sqft |\n|---|---|---|\n" + rows +
         f"\n\nThe spread between types is {(t['mean'].max() / t['mean'].min() - 1) * 100:.1f}%, so property "
         "type has little effect on price per sq ft in this dataset.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sample["Size_in_SqFt"], sample["Price_in_Lakhs"], s=4, alpha=0.25, color="#e76f51")
    ax.set(title="Q4. Size vs price (15k random sample)", xlabel="Size (sq ft)", ylabel="Price (lakhs)")
    r, txt = corr_text(d["Size_in_SqFt"], d["Price_in_Lakhs"], "size and price")
    rs = d["Size_in_SqFt"].corr(d["Price_in_Lakhs"], method="spearman")
    img = save(fig, "q04_size_vs_price")
    note("Q4. Is there a relationship between property size and price?", img,
         f"{txt} Spearman rho = {rs:+.3f}. Larger properties are **not** priced higher here, which is "
         "unrealistic for a real market and means price is driven by something other than size in this "
         "synthetic data (so price per sq ft is dominated by size: larger area -> lower price per sq ft).")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    sns.boxplot(y=d["Price_per_SqFt_INR_Raw"], ax=axes[0], color="#f4a261")
    axes[0].set(title="Price per sq ft (raw)", ylabel="INR per sq ft")
    sns.boxplot(y=d["Size_in_SqFt"], ax=axes[1], color="#2a9d8f")
    axes[1].set(title="Size")
    sns.boxplot(y=d["Price_in_Lakhs"], ax=axes[2], color="#3b6ea5")
    axes[2].set(title="Price (lakhs)")
    fig.suptitle("Q5. Outlier check (IQR rule)")
    n_ppsf = iqr_outliers(d["Price_per_SqFt_INR_Raw"])
    n_size = iqr_outliers(d["Size_in_SqFt"])
    n_price = iqr_outliers(d["Price_in_Lakhs"])
    img = save(fig, "q05_outliers")
    note("Q5. Are there any outliers in price per sq ft or property size?", img,
         f"IQR rule: **{n_ppsf:,}** price-per-sq-ft outliers ({n_ppsf / len(d) * 100:.1f}%), "
         f"**{n_size}** size outliers, **{n_price}** price outliers. The outliers in price per sq ft come "
         "from small, expensive listings. They are flagged (`PPSF_Outlier`) and the modelling feature is "
         "winsorised at the IQR fences.")


# ----------------------------------------------------------------------------
# 6-10 location
# ----------------------------------------------------------------------------
def q6_to_q10(d: pd.DataFrame) -> None:
    FINDINGS.append("## 6-10. Location-based Analysis\n")

    st = d.groupby("State")["Price_per_SqFt_INR_Raw"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(st.index, st.values, color=sns.color_palette(PAL, len(st)))
    ax.set(title="Q6. Average price per sq ft by state", xlabel="INR per sq ft")
    img = save(fig, "q06_ppsf_by_state")
    note("Q6. What is the average price per sq ft by state?", img,
         f"Highest: **{st.index[-1]}** (INR {st.iloc[-1]:,.0f}/sqft); lowest: **{st.index[0]}** "
         f"(INR {st.iloc[0]:,.0f}/sqft). Spread between states is only "
         f"{(st.iloc[-1] / st.iloc[0] - 1) * 100:.1f}%.")

    ct = d.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(ct.index, ct.values, color=sns.color_palette(PAL, len(ct)))
    ax.set(title="Q7. Average property price by city", xlabel="Price (lakhs)")
    ax.set_xlim(ct.min() * 0.97, ct.max() * 1.01)
    img = save(fig, "q07_price_by_city")
    note("Q7. What is the average property price by city?", img,
         f"Most expensive city: **{ct.index[-1]}** ({ct.iloc[-1]:.1f} lakhs); cheapest: **{ct.index[0]}** "
         f"({ct.iloc[0]:.1f} lakhs). All 42 cities average between {ct.min():.0f} and {ct.max():.0f} lakhs "
         "(axis truncated for readability) - city has almost no influence on price in this data.")

    loc = d.groupby("Locality")["Age_of_Property"].median().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.histplot(loc, bins=20, color="#6a4c93", ax=axes[0])
    axes[0].set(title="Median age across 500 localities", xlabel="Median age (years)")
    top = loc.tail(15)
    axes[1].barh(top.index, top.values, color="#6a4c93")
    axes[1].set(title="15 localities with the oldest median age", xlabel="Median age (years)")
    fig.suptitle("Q8. Median age of properties by locality")
    img = save(fig, "q08_median_age_by_locality")
    note("Q8. What is the median age of properties by locality?", img,
         f"Locality medians range from {loc.min():.0f} to {loc.max():.0f} years (overall median "
         f"{d['Age_of_Property'].median():.0f}). Oldest: **{loc.index[-1]}** ({loc.iloc[-1]:.0f} yrs); "
         f"newest: **{loc.index[0]}** ({loc.iloc[0]:.0f} yrs).")

    share = pd.crosstab(d["City"], d["BHK"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, 11))
    sns.heatmap(share, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "% of listings"}, ax=ax)
    ax.set(title="Q9. BHK distribution across cities (% of city listings)")
    img = save(fig, "q09_bhk_by_city")
    overall = d["BHK"].value_counts(normalize=True).sort_index() * 100
    note("Q9. How is BHK distributed across cities?", img,
         "Overall share: " + ", ".join(f"{k} BHK {v:.1f}%" for k, v in overall.items()) +
         ". Every city has a near-identical, roughly uniform mix of 1-5 BHK.")

    locp = d.groupby("Locality")["Price_in_Lakhs"].agg(["mean", "size"])
    top5 = locp.sort_values("mean", ascending=False).head(5).index.tolist()
    tmp = d[d["Locality"].isin(top5)].copy()
    tmp["Built_Period"] = pd.cut(tmp["Year_Built"], [1989, 1994, 1999, 2004, 2009, 2014, 2019, 2023],
                                 labels=["1990-94", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-23"])
    trend = tmp.groupby(["Built_Period", "Locality"], observed=True)["Price_in_Lakhs"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    trend.plot(marker="o", ax=ax)
    ax.set(title="Q10. Price trend (by build period) in the 5 most expensive localities",
           xlabel="Year built (5-year band)", ylabel="Average price (lakhs)")
    img = save(fig, "q10_top5_locality_trend")
    note("Q10. What are the price trends for the top 5 most expensive localities?", img,
         "Top 5 by mean price: " + ", ".join(f"**{k}** ({locp.loc[k, 'mean']:.1f} lakhs)" for k in top5) +
         ". The dataset has no listing date, so the trend is shown against construction period; the lines "
         "fluctuate without a consistent direction.")


# ----------------------------------------------------------------------------
# 11-15 relationships
# ----------------------------------------------------------------------------
def q11_to_q15(d: pd.DataFrame) -> None:
    FINDINGS.append("## 11-15. Feature Relationship & Correlation\n")

    cols = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR", "BHK", "Age_of_Property", "Floor_No",
            "Total_Floors", "Nearby_Schools", "Nearby_Hospitals", "Transport_Score", "Amenity_Count",
            "Infrastructure_Score"]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Q11. Correlation matrix of numeric features")
    img = save(fig, "q11_correlation_matrix")
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(key=np.abs, ascending=False)
    top = pairs.head(5)
    note("Q11. How are numeric features correlated with each other?", img,
         "Strongest pairs: " + "; ".join(f"{a} / {b} = {v:+.2f}" for (a, b), v in top.items()) +
         ". Price per sq ft is strongly (negatively) tied to size and (positively) to price because it is "
         "derived from them; the Infrastructure_Score pairs and the Floor_No / Total_Floors pair are likewise "
         "by construction (the score is built from schools and transport; the floor fix puts the smaller of "
         "the two raw values in Floor_No). Apart from such derived features, raw features are essentially "
         "independent.")

    for qn, col, label in [(12, "Nearby_Schools", "schools"), (13, "Nearby_Hospitals", "hospitals")]:
        g = d.groupby(col)["Price_per_SqFt_INR"].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(g.index, g.values, marker="o", color="#e63946")
        ax.set(title=f"Q{qn}. Nearby {label} vs average price per sq ft", xlabel=f"Number of nearby {label}",
               ylabel="INR per sq ft")
        r, txt = corr_text(d[col], d["Price_per_SqFt_INR"], f"nearby {label} and price per sq ft")
        img = save(fig, f"q{qn}_{label}_vs_ppsf")
        note(f"Q{qn}. How do nearby {label} relate to price per sq ft?", img,
             f"{txt} Average price per sq ft moves between INR {g.min():,.0f} and INR {g.max():,.0f} across "
             f"1-10 {label}, with no systematic trend.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=d, x="Furnished_Status", y="Price_in_Lakhs", hue="Furnished_Status", palette=PAL,
                legend=False, ax=axes[0])
    sns.barplot(data=d, x="Furnished_Status", y="Price_per_SqFt_INR", hue="Furnished_Status", palette=PAL,
                legend=False, ax=axes[1])
    axes[0].set(title="Price (lakhs)")
    axes[1].set(title="Avg price per sq ft (INR)")
    fig.suptitle("Q14. Price by furnished status")
    f = d.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
    img = save(fig, "q14_price_by_furnished")
    note("Q14. How does price vary by furnished status?", img,
         "Average price: " + ", ".join(f"{k} {v:.1f} lakhs" for k, v in f.items()) +
         f". Differences are within {(f.max() / f.min() - 1) * 100:.1f}%.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=d, x="Facing", y="Price_per_SqFt_INR", hue="Facing", palette=PAL, legend=False, ax=ax)
    ax.set(title="Q15. Price per sq ft by facing direction", ylabel="INR per sq ft")
    f = d.groupby("Facing")["Price_per_SqFt_INR"].mean()
    img = save(fig, "q15_ppsf_by_facing")
    note("Q15. How does price per sq ft vary by property facing direction?", img,
         "Average INR/sqft: " + ", ".join(f"{k} {v:,.0f}" for k, v in f.items()) +
         f". Best: **{f.idxmax()}**, but the gap to the lowest is only {(f.max() / f.min() - 1) * 100:.1f}%.")


# ----------------------------------------------------------------------------
# 16-20 investment / amenities / ownership
# ----------------------------------------------------------------------------
def q16_to_q20(d: pd.DataFrame) -> None:
    FINDINGS.append("## 16-20. Investment / Amenities / Ownership Analysis\n")

    oc = d["Owner_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(oc.index, oc.values, color=sns.color_palette(PAL, len(oc)))
    for i, v in enumerate(oc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set(title="Q16. Properties per owner type", ylabel="Listings")
    img = save(fig, "q16_owner_type")
    note("Q16. How many properties belong to each owner type?", img,
         "; ".join(f"**{k}**: {v:,} ({v / len(d) * 100:.1f}%)" for k, v in oc.items()) + ".")

    av = d["Availability_Status"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(av.index, av.values, color=sns.color_palette(PAL, len(av)))
    for i, v in enumerate(av.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    ax.set(title="Q17. Properties per availability status", ylabel="Listings")
    img = save(fig, "q17_availability")
    note("Q17. How many properties are available under each availability status?", img,
         "; ".join(f"**{k}**: {v:,} ({v / len(d) * 100:.1f}%)" for k, v in av.items()) +
         ". (The data has only these two statuses; there is no 'Sold' category.)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sns.barplot(data=d, x="Parking_Space", y="Price_in_Lakhs", hue="Parking_Space", palette=PAL, legend=False, ax=axes[0])
    sns.barplot(data=d, x="Parking_Space", y="Price_per_SqFt_INR", hue="Parking_Space", palette=PAL, legend=False, ax=axes[1])
    axes[0].set(title="Avg price (lakhs)")
    axes[1].set(title="Avg price per sq ft (INR)")
    fig.suptitle("Q18. Does parking space affect price?")
    p = d.groupby("Parking_Space")["Price_in_Lakhs"].mean()
    img = save(fig, "q18_parking_vs_price")
    note("Q18. Does parking space affect property price?", img,
         f"With parking: {p.get('Yes', np.nan):.1f} lakhs; without: {p.get('No', np.nan):.1f} lakhs "
         f"(difference {(p.get('Yes', np.nan) / p.get('No', np.nan) - 1) * 100:+.2f}%). Parking has no "
         "meaningful effect on price in this dataset.")

    rows = []
    for a in C.AMENITY_LIST:
        m = d.groupby(f"Has_{a}")["Price_per_SqFt_INR"].mean()
        rows.append((a, m.get(0, np.nan), m.get(1, np.nan)))
    am = pd.DataFrame(rows, columns=["Amenity", "Without", "With"]).set_index("Amenity")
    cnt = d.groupby("Amenity_Count")["Price_per_SqFt_INR"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    am.plot.bar(ax=axes[0], color=["#adb5bd", "#2a9d8f"], rot=0)
    axes[0].set(title="Avg price per sq ft with / without amenity", ylabel="INR per sq ft")
    axes[1].plot(cnt.index, cnt.values, marker="o", color="#e76f51")
    axes[1].set(title="By number of amenities", xlabel="Amenity count", ylabel="INR per sq ft")
    fig.suptitle("Q19. How do amenities affect price per sq ft?")
    img = save(fig, "q19_amenities_vs_ppsf")
    r, txt = corr_text(d["Amenity_Count"], d["Price_per_SqFt_INR"], "amenity count and price per sq ft")
    best = (am["With"] / am["Without"] - 1).abs().idxmax()
    note("Q19. How do amenities affect price per sq ft?", img,
         f"{txt} The largest with/without gap is for **{best}** "
         f"({(am.loc[best, 'With'] / am.loc[best, 'Without'] - 1) * 100:+.1f}%) - negligible in practice.")

    g = d.groupby("Public_Transport_Accessibility").agg(
        ppsf=("Price_per_SqFt_INR", "mean"), good=(C.CLF_TARGET, "mean"), growth=("Annual_Growth_Rate", "mean"))
    g = g.loc[["Low", "Medium", "High"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(g.index, g["ppsf"], color=sns.color_palette(PAL, 3))
    axes[0].set(title="Avg price per sq ft (INR)")
    axes[1].bar(g.index, g["good"] * 100, color=sns.color_palette(PAL, 3))
    axes[1].set(title="% Good Investment")
    axes[2].bar(g.index, g["growth"] * 100, color=sns.color_palette(PAL, 3))
    axes[2].set(title="Avg modelled annual growth (%)")
    fig.suptitle("Q20. Public transport accessibility vs price and investment potential")
    img = save(fig, "q20_transport")
    note("Q20. How does public transport accessibility relate to price per sq ft or investment potential?", img,
         "| Accessibility | Avg INR/sqft | % Good investment | Avg annual growth |\n|---|---|---|---|\n" +
         "\n".join(f"| {i} | {r['ppsf']:,.0f} | {r['good'] * 100:.1f}% | {r['growth'] * 100:.2f}% |" for i, r in g.iterrows()) +
         "\n\nTransport access has no effect on price per sq ft, but it clearly raises investment potential "
         "(by construction of the growth model and investment score).")


# ----------------------------------------------------------------------------
# Step-2 extras
# ----------------------------------------------------------------------------
def step2_extras(d: pd.DataFrame) -> None:
    FINDINGS.append("## Additional Step-2 Analyses\n")

    top_cities = d.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(6).index
    tmp = d[d["City"].isin(top_cities)].copy()
    tmp["Built_Period"] = pd.cut(tmp["Year_Built"], [1989, 1994, 1999, 2004, 2009, 2014, 2019, 2023],
                                 labels=["1990-94", "1995-99", "2000-04", "2005-09", "2010-14", "2015-19", "2020-23"])
    tr = tmp.groupby(["Built_Period", "City"], observed=True)["Price_in_Lakhs"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    tr.plot(marker="o", ax=ax)
    ax.set(title="Price trends by city (6 highest-priced cities, by build period)", xlabel="Year built",
           ylabel="Average price (lakhs)")
    img = save(fig, "x1_price_trends_by_city")
    gr = d.groupby("City")["Annual_Growth_Rate"].mean().sort_values(ascending=False)
    note("Price trends by city", img,
         f"Highest modelled growth: **{gr.index[0]}** ({gr.iloc[0] * 100:.2f}%/yr); lowest: **{gr.index[-1]}** "
         f"({gr.iloc[-1] * 100:.2f}%/yr). Current prices show no consistent trend over construction period.")

    d2 = d.assign(Size_Band=pd.cut(d["Size_in_SqFt"], [0, 1000, 2000, 3000, 4000, 5000],
                                   labels=["<1k", "1-2k", "2-3k", "3-4k", "4-5k"]))
    gs = d2.groupby("Size_Band", observed=True).agg(
        roi=("Appreciation_Pct_5Y", "mean"), good=(C.CLF_TARGET, "mean"), ppsf=("Price_per_SqFt_INR", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(gs.index.astype(str), gs["roi"], color="#2a9d8f")
    axes[0].set(title="Avg 5-year appreciation (%)", xlabel="Size band (sq ft)")
    axes[1].bar(gs.index.astype(str), gs["good"] * 100, color="#e9c46a")
    axes[1].set(title="% Good Investment", xlabel="Size band (sq ft)")
    fig.suptitle("Area vs investment return")
    r, txt = corr_text(d["Size_in_SqFt"], d["Appreciation_Pct_5Y"], "size and 5-year appreciation")
    r2, txt2 = corr_text(d["Size_in_SqFt"], d[C.CLF_TARGET], "size and Good Investment")
    img = save(fig, "x2_area_vs_return")
    note("Correlation between area and investment return", img,
         f"{txt} {txt2} Larger homes have a lower price per sq ft, so they score better on the "
         "'value' component of the investment score, which is why size relates to the Good Investment label "
         "even though it does not relate to appreciation.")

    sec = d.groupby("Security")[C.CLF_TARGET].mean() * 100
    d3 = d.assign(Infra_Band=pd.cut(d["Infrastructure_Score"], [0, 3, 4, 5, 6, 7, 10]))
    gi = d3.groupby("Infra_Band", observed=True)[C.CLF_TARGET].mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(sec.index, sec.values, color=["#adb5bd", "#2a9d8f"])
    axes[0].set(title="% Good Investment by security", ylabel="%")
    axes[1].bar(gi.index.astype(str), gi.values, color=sns.color_palette(PAL, len(gi)))
    axes[1].set(title="% Good Investment by infrastructure score band", ylabel="%")
    fig.suptitle("Safety proxy vs Good Investment")
    img = save(fig, "x3_safety_vs_investment")
    note("Impact of crime rate on Good Investment classification (proxy)", img,
         "**The dataset has no crime-rate column**, so security features and the infrastructure score are "
         f"used as a safety proxy. Good-investment rate: {sec.get('Yes', np.nan):.1f}% with security vs "
         f"{sec.get('No', np.nan):.1f}% without; it rises from {gi.iloc[0]:.1f}% to {gi.iloc[-1]:.1f}% "
         "across infrastructure bands.")

    gb = d3.groupby("Infra_Band", observed=True).agg(price=("Price_in_Lakhs", "mean"), future=(C.REG_TARGET, "mean"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(gb))
    ax.bar(x - 0.2, gb["price"], 0.4, label="Current price", color="#3b6ea5")
    ax.bar(x + 0.2, gb["future"], 0.4, label="Price after 5 years", color="#e76f51")
    ax.set_xticks(x, gb.index.astype(str))
    ax.set(title="Infrastructure score vs resale value", xlabel="Infrastructure score band", ylabel="Lakhs")
    ax.legend()
    r, txt = corr_text(d["Infrastructure_Score"], d[C.REG_TARGET], "infrastructure score and 5-year resale value")
    r2, txt2 = corr_text(d["Infrastructure_Score"], d["Price_in_Lakhs"], "infrastructure score and current price")
    img = save(fig, "x4_infrastructure_vs_resale")
    note("Relationship between infrastructure score and resale value", img,
         f"{txt2} {txt} Better infrastructure raises the modelled growth rate only slightly, so resale "
         "value is almost unchanged.")


def run() -> None:
    d = pd.read_csv(C.DATA_PROCESSED)
    sample = d.sample(15000, random_state=C.RANDOM_STATE)
    FINDINGS.clear()
    FINDINGS.append(
        "# EDA Findings\n\n"
        f"Dataset: **{len(d):,} listings, {d['City'].nunique()} cities, {d['State'].nunique()} states, "
        f"{d['Locality'].nunique()} localities**. Prices are in lakhs of INR; price per sq ft is in INR.\n\n"
        "> **Important context:** the supplied data behaves like a synthetic dataset - most raw features are "
        "statistically independent of price, so many relationships below are weak or absent. The analysis "
        "reports what the data shows rather than what a real market would be expected to show.\n"
    )
    q1_to_q5(d, sample)
    q6_to_q10(d)
    q11_to_q15(d)
    q16_to_q20(d)
    step2_extras(d)
    (C.REPORTS_DIR / "eda_findings.md").write_text("\n".join(FINDINGS), encoding="utf-8")
    print(f"EDA complete: {len(list(C.EDA_DIR.glob('*.png')))} charts -> {C.EDA_DIR}")


if __name__ == "__main__":
    run()
