import pandas as pd
import streamlit as st
import altair as alt

# Config
st.set_page_config(page_title="ML Forecasting Case Study", layout="wide")

# Load data
rb = pd.read_csv("data/predictions/rb_predictions.csv")
ml = pd.read_csv("data/predictions/ml_predictions.csv")
sales = pd.read_csv("data/processed/sales.csv")
importance = pd.read_csv("data/predictions/avg_feature_importance.csv")


# Convert YEAR and WEEKNUM into datetime
def year_week_to_date(df):
    return pd.to_datetime(
        df["YEAR"].astype(str) + df["WEEKNUM"].astype(str).str.zfill(2) + "1",
        format="%G%V%u",
    )


rb["date"] = year_week_to_date(rb)
ml["date"] = year_week_to_date(ml)
sales["date"] = year_week_to_date(sales)

# Rename columns for clarity
rb = rb.rename(
    columns={
        "SeasonalSummer": "SeasonalSummer (Ratio Based)",
        "TrendingChristmas": "TrendingChristmas (Ratio Based)",
        "SeasonalWinter": "SeasonalWinter (Ratio Based)",
    }
)
ml = ml.rename(
    columns={
        "SeasonalSummer": "SeasonalSummer (ML Model)",
        "TrendingChristmas": "TrendingChristmas (ML Model)",
        "SeasonalWinter": "SeasonalWinter (ML Model)",
    }
)

# Setup
categories = ["SeasonalSummer", "TrendingChristmas", "SeasonalWinter"]
category_labels = {
    "SeasonalSummer": "Summer-Driven Demand",
    "TrendingChristmas": "Holiday Peaks",
    "SeasonalWinter": "Trending Demand",
}

# Page title and intro
st.title("ML Forecasting Case Study")

layout_cols = st.columns([1, 3])

# Left: Description column
with layout_cols[0]:
    st.markdown("### Forecast vs Actual")
    st.markdown(
        "This chart compares actual weekly demand with two predictive approaches: "
        "a simple **ratio-based model** that extrapolates from prior-year sales patterns, "
        "and a machine learning model trained on lagged signals, trends, and seasonality."
    )

    st.markdown(
        "**Ratio-based model:**  $\\hat{y}_t = y_{t-1} \\times \\frac{y'_{t}}{y'_{t-1}}$"
    )
    st.caption("Where $y'$ refers to sales in the same period one year ago.")

    st.markdown("**Variables used by the ML model**")
    st.markdown(
        "The ML model uses a combination of:\n"
        "- **Lag features**, such as previous week's demand ($y_{t-1}$), to capture short-term trends.\n"
        "- **Rolling averages**, like the 4- or 8-week moving average, to smooth out short-term fluctuations.\n"
        "- **Fourier terms**, which encode seasonality using repeating sine and cosine signals.\n"
        "- **Calendar features**, such as the week of the year, to align patterns with seasonal effects."
    )


# Right: Graph section with 3 rows
with layout_cols[1]:
    for category in categories:
        st.markdown(f"#### {category_labels[category]}")

        # Prepare data
        df = sales[["date", category]].rename(columns={category: "Actual"})
        df = df.merge(rb[["date", f"{category} (Ratio Based)"]], on="date", how="outer")
        df = df.merge(ml[["date", f"{category} (ML Model)"]], on="date", how="outer")
        df = df.sort_values("date")

        # Rename columns to generic labels for consistent legends
        df = df.rename(
            columns={
                f"{category} (Ratio Based)": "Ratio Based",
                f"{category} (ML Model)": "ML Model",
            }
        )

        # Split: 2/3 chart, 1/3 feature importance
        chart_col, feature_col = st.columns([2, 1])

        with chart_col:
            chart = (
                alt.Chart(df)
                .transform_fold(
                    ["Actual", "Ratio Based", "ML Model"], as_=["Model", "Value"]
                )
                .mark_line()
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("Value:Q", title=None),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(
                            domain=["Actual", "Ratio Based", "ML Model"],
                            range=["lightgray", "blue", "black"],
                        ),
                    ),
                    strokeDash=alt.StrokeDash(
                        "Model:N",
                        scale=alt.Scale(
                            domain=["Actual", "Ratio Based", "ML Model"],
                            range=[[1, 0], [2, 2], [1, 0]],
                        ),
                    ),
                )
                .properties(height=200)
            )
            st.altair_chart(chart, use_container_width=True)

        with feature_col:
            top_features = (
                importance[importance["category"] == category]
                .groupby("feature", as_index=False)["mean_importance"]
                .mean()
                .sort_values("mean_importance", ascending=False)
                .head(7)
            )

            bar_chart = (
                alt.Chart(top_features)
                .mark_bar()
                .encode(
                    x=alt.X("mean_importance:Q", title=None),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.value("black"),
                )
                .properties(height=200)
            )

            st.altair_chart(bar_chart, use_container_width=True)
