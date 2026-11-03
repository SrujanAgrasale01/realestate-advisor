"""Step 5 - Streamlit app: Real Estate Investment Advisor.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config as C  # noqa: E402
from src.predict import Advisor, format_lakhs  # noqa: E402

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

PAGES = ["🔮 Investment Advisor", "🔎 Property Explorer", "📊 Market Insights", "🧪 Model Performance"]
DATA_COLUMNS = [
    "ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
    "Price_per_SqFt_INR", "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Facing", "Owner_Type", "Availability_Status", "Amenity_Count",
    "Infrastructure_Score", "Annual_Growth_Rate", "Future_Price_5Y", "Appreciation_Pct_5Y",
    "Good_Investment",
]
CATEGORY_COLUMNS = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Facing", "Owner_Type",
    "Availability_Status",
]


# ----------------------------------------------------------------------------
# cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def get_advisor() -> Advisor:
    return Advisor()


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_PROCESSED, usecols=DATA_COLUMNS)
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")
    return df


@st.cache_data
def load_metrics() -> dict:
    return json.loads(C.METRICS_JSON.read_text())


@st.cache_data
def load_importance(kind: str) -> pd.DataFrame:
    path = C.CLF_IMPORTANCE_CSV if kind == "clf" else C.REG_IMPORTANCE_CSV
    return pd.read_csv(path)


METRIC_CHOICES = {
    "Average price per sq ft (INR)": ("Price_per_SqFt_INR", "mean", 1),
    "Average price (lakhs)": ("Price_in_Lakhs", "mean", 1),
    "Average modelled annual growth (%)": ("Annual_Growth_Rate", "mean", 100),
    "Good-investment rate (%)": ("Good_Investment", "mean", 100),
    "Average 5-year appreciation (%)": ("Appreciation_Pct_5Y", "mean", 1),
}


@st.cache_data
def pivot_metric(level: str, columns: str, metric: str) -> pd.DataFrame:
    col, fn, scale = METRIC_CHOICES[metric]
    pivot = load_data().pivot_table(index=level, columns=columns, values=col, aggfunc=fn, observed=True)
    pivot.columns = [str(c) for c in pivot.columns]
    return (pivot * scale).round(2)


@st.cache_data
def trend_table(cities: tuple, value: str) -> pd.DataFrame:
    df = load_data()
    sub = df[df["City"].isin(cities)]
    out = sub.groupby(["Year_Built", "City"], observed=True)[value].mean().reset_index()
    out["City"] = out["City"].astype(str)
    return out


@st.cache_data
def correlation_matrix() -> pd.DataFrame:
    cols = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR", "BHK", "Age_of_Property",
            "Nearby_Schools", "Nearby_Hospitals", "Infrastructure_Score", "Amenity_Count",
            "Annual_Growth_Rate", "Future_Price_5Y", "Good_Investment"]
    return load_data()[cols].corr().round(2)


# ----------------------------------------------------------------------------
# page 1 - advisor
# ----------------------------------------------------------------------------
def page_advisor() -> None:
    adv = get_advisor()
    opt = adv.options
    st.title("🏠 Real Estate Investment Advisor")
    st.caption("Enter a property's details to find out whether it is a good investment and what it "
               "could be worth in 5 years.")

    st.subheader("1. Location")
    lc1, lc2 = st.columns(2)
    state = lc1.selectbox("State", sorted(opt["states"]), index=sorted(opt["states"]).index("Maharashtra")
                          if "Maharashtra" in opt["states"] else 0)
    city = lc2.selectbox("City", opt["states"][state])

    st.subheader("2. Property details")
    r, d = opt["ranges"], opt["defaults"]
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            property_type = st.selectbox("Property type", opt["property_types"])
            bhk = st.slider("BHK", r["bhk"][0], r["bhk"][1], 3)
            size = st.number_input("Size (sq ft)", min_value=100, max_value=20000, value=d["size"], step=50)
            price = st.number_input("Current price (₹ lakhs)", min_value=1.0, max_value=5000.0,
                                    value=d["price"], step=5.0)
            owner = st.selectbox("Owner type", opt["owner_types"])
        with c2:
            year_built = st.number_input("Year built", min_value=1950, max_value=C.CURRENT_YEAR,
                                         value=d["year_built"], step=1)
            furnished = st.selectbox("Furnished status", opt["furnished_status"])
            total_floors = st.number_input("Total floors in building", min_value=1, max_value=100, value=10, step=1)
            floor_no = st.number_input("Floor number", min_value=0, max_value=100, value=3, step=1)
            facing = st.selectbox("Facing", opt["facing"])
        with c3:
            schools = st.slider("Nearby schools", r["schools"][0], r["schools"][1], 5)
            hospitals = st.slider("Nearby hospitals", r["hospitals"][0], r["hospitals"][1], 5)
            transport = st.selectbox("Public transport accessibility", opt["transport"], index=1)
            availability = st.selectbox("Availability status", opt["availability"])
            pc, sc = st.columns(2)
            parking = pc.radio("Parking", ["Yes", "No"], horizontal=True)
            security = sc.radio("Security", ["Yes", "No"], horizontal=True)
        amenities = st.multiselect("Amenities", opt["amenities"], default=["Gym", "Garden"])
        submitted = st.form_submit_button("🔍 Analyze investment", type="primary")

    if submitted:
        inputs = {
            "State": state, "City": city, "Property_Type": property_type, "BHK": int(bhk),
            "Size_in_SqFt": float(size), "Price_in_Lakhs": float(price), "Year_Built": int(year_built),
            "Furnished_Status": furnished, "Floor_No": int(floor_no), "Total_Floors": int(total_floors),
            "Nearby_Schools": int(schools), "Nearby_Hospitals": int(hospitals),
            "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
            "Facing": facing, "Owner_Type": owner, "Availability_Status": availability,
            "Amenities": amenities,
        }
        try:
            st.session_state["advisor_result"] = (inputs, adv.predict(inputs))
        except ValueError as exc:
            st.session_state.pop("advisor_result", None)
            st.error(str(exc))

    if "advisor_result" in st.session_state:
        show_result(*st.session_state["advisor_result"])


def show_result(inputs: dict, res: dict) -> None:
    st.divider()
    st.subheader("3. Result")
    r = get_advisor().options["ranges"]
    if not (r["price"][0] <= inputs["Price_in_Lakhs"] <= r["price"][1]) or not (
            r["size"][0] <= inputs["Size_in_SqFt"] <= r["size"][1]):
        st.info("Price or size is outside the range of the training data "
                f"({r['price'][0]:.0f}-{r['price'][1]:.0f} lakhs, {r['size'][0]}-{r['size'][1]} sq ft); "
                "treat the estimate with extra caution.")

    if res["is_good"]:
        st.success(f"✅ **Good Investment** - the model is {res['confidence'] * 100:.1f}% confident.")
    else:
        st.warning(f"⚠️ **Not a Good Investment** - the model is {res['confidence'] * 100:.1f}% confident.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated price after 5 years", format_lakhs(res["future_price"]),
              f"{res['appreciation_pct']:+.1f}% vs today")
    m2.metric("Implied annual growth", f"{res['cagr_pct']:.2f}% p.a.")
    m3.metric("Price per sq ft", f"₹ {res['price_per_sqft']:,.0f}",
              f"{(res['price_per_sqft'] / res['city_median_ppsf'] - 1) * 100:+.0f}% vs {inputs['City']} median",
              delta_color="inverse")
    m4.metric("Infrastructure score", f"{res['infrastructure_score']:.1f} / 10")

    tab1, tab2, tab3 = st.tabs(["Confidence", "5-year projection", "Feature importance"])
    with tab1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=res["prob_good"] * 100, number={"suffix": "%"},
            title={"text": "Probability of being a Good Investment"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2a9d8f" if res["is_good"] else "#e76f51"},
                   "steps": [{"range": [0, 50], "color": "#fde8e4"}, {"range": [50, 100], "color": "#e0f3ef"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
        gauge.update_layout(height=320, margin=dict(t=60, b=10))
        cmp = pd.DataFrame({"Measure": [f"Your property", f"{inputs['City']} median"],
                            "INR per sq ft": [res["price_per_sqft"], res["city_median_ppsf"]]})
        g1, g2 = st.columns(2)
        g1.plotly_chart(gauge, width="stretch")
        bar = px.bar(cmp, x="Measure", y="INR per sq ft", color="Measure", text_auto=",.0f",
                     title="Price per sq ft vs city median")
        bar.update_layout(height=320, showlegend=False, margin=dict(t=60, b=10))
        g2.plotly_chart(bar, width="stretch")
    with tab2:
        years = list(range(0, C.HORIZON_YEARS + 1))
        price = inputs["Price_in_Lakhs"]
        cagr = res["cagr_pct"] / 100
        proj = pd.DataFrame({
            "Year": years * 2,
            "Value (lakhs)": [price * (1 + cagr) ** y for y in years] + [price * 1.08 ** y for y in years],
            "Scenario": ["Model estimate"] * len(years) + ["Fixed 8% benchmark"] * len(years),
        })
        fig = px.line(proj, x="Year", y="Value (lakhs)", color="Scenario", markers=True,
                      title="Projected property value (₹ lakhs)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with tab3:
        f1, f2 = st.columns(2)
        for col, kind, title in [(f1, "clf", "What drives the Good Investment verdict"),
                                 (f2, "reg", "What drives the 5-year price estimate")]:
            imp = load_importance(kind).head(10).iloc[::-1]
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title=title)
            fig.update_layout(height=380, xaxis_tickformat=".0%", yaxis_title=None)
            col.plotly_chart(fig, width="stretch")
        st.caption("Permutation importance on the held-out test set: how much model accuracy drops when a "
                   "feature is shuffled (shares sum to 100%). It describes the model overall, not this one property.")


# ----------------------------------------------------------------------------
# page 2 - explorer
# ----------------------------------------------------------------------------
def page_explorer() -> None:
    df = load_data()
    st.title("🔎 Property Explorer")
    st.caption("Filter the 250,000 listings by location, area, price, BHK and more.")

    f1, f2, f3 = st.columns(3)
    states = f1.multiselect("State", sorted(df["State"].cat.categories))
    city_pool = sorted(df.loc[df["State"].isin(states), "City"].unique()) if states else sorted(df["City"].cat.categories)
    cities = f2.multiselect("City", city_pool)
    types = f3.multiselect("Property type", sorted(df["Property_Type"].cat.categories))
    f4, f5, f6 = st.columns(3)
    bhks = f4.multiselect("BHK", sorted(df["BHK"].unique()))
    avail = f5.multiselect("Availability", sorted(df["Availability_Status"].cat.categories))
    good_only = f6.checkbox("Only rule-based 'Good Investment' listings")
    s1, s2 = st.columns(2)
    size_rng = s1.slider("Area (sq ft)", int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max()),
                         (int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())), step=50)
    price_rng = s2.slider("Price (₹ lakhs)", float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()),
                          (float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())), step=5.0)

    mask = df["Size_in_SqFt"].between(*size_rng) & df["Price_in_Lakhs"].between(*price_rng)
    if states:
        mask &= df["State"].isin(states)
    if cities:
        mask &= df["City"].isin(cities)
    if types:
        mask &= df["Property_Type"].isin(types)
    if bhks:
        mask &= df["BHK"].isin(bhks)
    if avail:
        mask &= df["Availability_Status"].isin(avail)
    if good_only:
        mask &= df["Good_Investment"] == 1
    view = df[mask]

    if view.empty:
        st.warning("No properties match these filters. Try widening them.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching listings", f"{len(view):,}")
    k2.metric("Average price", format_lakhs(view["Price_in_Lakhs"].mean()))
    k3.metric("Average price per sq ft", f"₹ {view['Price_per_SqFt_INR'].mean():,.0f}")
    k4.metric("Good-investment share", f"{view['Good_Investment'].mean() * 100:.1f}%")

    sort_col = st.selectbox("Sort by", ["Price_in_Lakhs", "Price_per_SqFt_INR", "Size_in_SqFt",
                                        "Appreciation_Pct_5Y", "Age_of_Property"])
    ascending = st.radio("Order", ["Ascending", "Descending"], horizontal=True, index=1) == "Ascending"
    show_cols = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                 "Price_per_SqFt_INR", "Year_Built", "Availability_Status", "Future_Price_5Y", "Good_Investment"]
    table = view.sort_values(sort_col, ascending=ascending).head(20000)[show_cols].copy()
    table["Good_Investment"] = np.where(table["Good_Investment"] == 1, "Yes", "No")
    table = table.rename(columns={"Future_Price_5Y": "Future_Price_5Y_(rule)", "Good_Investment": "Good_(rule)"})
    st.dataframe(table.head(1000).round(2), width="stretch", hide_index=True)
    st.caption(f"Showing the first 1,000 of {len(view):,} matching rows; the download holds up to 20,000 rows.")
    st.download_button("⬇️ Download filtered rows (CSV)", table.round(2).to_csv(index=False).encode("utf-8"),
                       file_name="filtered_properties.csv", mime="text/csv")

    c1, c2 = st.columns(2)
    sample = view.sample(min(len(view), 30000), random_state=C.RANDOM_STATE)
    c1.plotly_chart(px.histogram(sample, x="Price_in_Lakhs", nbins=40, title="Price distribution (lakhs)"),
                    width="stretch")
    city_avg = (view.groupby("City", observed=True)["Price_per_SqFt_INR"].mean().sort_values(ascending=False)
                .head(15).reset_index())
    city_avg["City"] = city_avg["City"].astype(str)
    c2.plotly_chart(px.bar(city_avg, x="City", y="Price_per_SqFt_INR", title="Average price per sq ft by city (top 15)"),
                    width="stretch")


# ----------------------------------------------------------------------------
# page 3 - insights
# ----------------------------------------------------------------------------
def page_insights() -> None:
    df = load_data()
    st.title("📊 Market Insights")

    st.subheader("Location-wise heatmap")
    h1, h2, h3 = st.columns(3)
    metric = h1.selectbox("Metric", list(METRIC_CHOICES))
    level = h2.selectbox("Rows", ["City", "State"])
    columns = h3.selectbox("Columns", ["Property_Type", "BHK", "Availability_Status", "Furnished_Status"])
    pivot = pivot_metric(level, columns, metric)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlGnBu", text_auto=".0f" if "INR" in metric else ".1f",
                    labels={"color": metric})
    fig.update_layout(height=max(380, 18 * len(pivot) + 120))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Trend charts")
    top_cities = df.groupby("City", observed=True)["Price_in_Lakhs"].mean().sort_values(ascending=False).head(3).index
    chosen = st.multiselect("Cities to compare", sorted(df["City"].cat.categories),
                            default=[str(c) for c in top_cities], max_selections=6)
    value = st.radio("Value", ["Price_in_Lakhs", "Future_Price_5Y", "Price_per_SqFt_INR"], horizontal=True)
    if chosen:
        trend = trend_table(tuple(chosen), value)
        fig = px.line(trend, x="Year_Built", y=value, color="City", markers=True,
                      title=f"Average {value.replace('_', ' ')} by construction year")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pick at least one city to see its trend.")

    c1, c2 = st.columns(2)
    growth = (df.groupby("City", observed=True)["Annual_Growth_Rate"].mean().sort_values(ascending=False)
              .head(12).mul(100).reset_index())
    growth["City"] = growth["City"].astype(str)
    c1.plotly_chart(px.bar(growth, x="City", y="Annual_Growth_Rate", title="Top 12 cities by modelled annual growth (%)"),
                    width="stretch")
    corr = correlation_matrix()
    c2.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation matrix",
                              aspect="auto"), width="stretch")
    st.caption("Note: the supplied dataset is largely synthetic, so many raw features (size, city, schools, "
               "amenities...) show almost no relationship with price. See reports/eda_findings.md for details.")


# ----------------------------------------------------------------------------
# page 4 - model performance
# ----------------------------------------------------------------------------
def page_models() -> None:
    m = load_metrics()
    best = m["best"]
    st.title("🧪 Model Performance")
    ds = m["dataset"]
    st.caption(f"{ds['train_rows']:,} training rows / {ds['test_rows']:,} held-out test rows. "
               f"Champion models: **{best['classifier']}** (classification) and **{best['regressor']}** (regression).")

    st.subheader("Classification - Good Investment")
    clf = pd.DataFrame(m["classification"]).T[["accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"]]
    st.dataframe(clf.round(4), width="stretch")
    cm = np.array(m["classification"][best["classifier"]]["confusion_matrix"])
    c1, c2 = st.columns(2)
    fig = px.imshow(cm, text_auto=",", x=["Pred: Not good", "Pred: Good"], y=["Actual: Not good", "Actual: Good"],
                    color_continuous_scale="Blues", title=f"Confusion matrix - {best['classifier']}")
    c1.plotly_chart(fig, width="stretch")
    roc = m["roc_curve"]
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Regression - Price after 5 years")
    reg = pd.DataFrame(m["regression"]).T[["rmse", "mae", "r2", "mape_pct", "train_seconds"]]
    st.dataframe(reg.round(4), width="stretch")
    pva = pd.DataFrame(m["pred_vs_actual"])
    fig = px.scatter(pva, x="actual", y="predicted", opacity=0.4, title=f"Predicted vs actual - {best['regressor']}",
                     labels={"actual": "Actual (lakhs)", "predicted": "Predicted (lakhs)"})
    lim = [float(pva.min().min()), float(pva.max().max())]
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Perfect", line=dict(dash="dash", color="black"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Feature importance")
    f1, f2 = st.columns(2)
    for col, kind, title in [(f1, "clf", "Classifier"), (f2, "reg", "Regressor")]:
        imp = load_importance(kind).head(15).iloc[::-1]
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title=title)
        fig.update_layout(xaxis_tickformat=".0%", yaxis_title=None)
        col.plotly_chart(fig, width="stretch")
    st.caption("Permutation importance on the held-out test set (share of total accuracy loss when a feature is "
               "shuffled). Correlated features such as price, log-price and price per sq ft share credit.")

    with st.expander("How the targets are defined / MLflow"):
        st.markdown(
            f"""
**Future price** = current price × (1 + r)^5, where the annual rate *r* is feature-based (city tier, property
type, availability, transport access, infrastructure, building age, plus small random noise).

**Good Investment** = 1 when a 0-100 multi-factor score ≥ {C.GOOD_INVESTMENT_THRESHOLD}
(value vs city median price/sq ft, modelled growth, BHK ≥ 3, ready-to-move, transport, schools & hospitals,
security, parking, amenities).

Both targets are rule-derived because the dataset has no historical or future prices, so the metrics above
measure how well the models learn these rules - not real-market forecasting accuracy.

**Experiment tracking:** `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://localhost:5000.
Registered models: `{m['registry']['classifier']['registered_name']}` and
`{m['registry']['regressor']['registered_name']}` (alias `champion`).
"""
        )


# ----------------------------------------------------------------------------
def main() -> None:
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Real Estate Investment Advisor\nPredicting property profitability & future value")
    if page == PAGES[0]:
        page_advisor()
    elif page == PAGES[1]:
        page_explorer()
    elif page == PAGES[2]:
        page_insights()
    else:
        page_models()


main()
