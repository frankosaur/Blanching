import sys
import streamlit as st

st.title("Blanching Calculator")
st.write(
    "Predict your blanching time based on past data. Help keep this updated by adding your data!"
)
st.markdown(
    """
    <h3 style="color:#F04A00;">Calculate Time</h3>
    </div>
""",
unsafe_allow_html=True
)

import pandas as pd

df = pd.read_csv("blanch_data.csv")

bean = st.selectbox(
"Ingredient Number",
sorted(df["Ingredient Number"].dropna().unique())
)

moisture_target = st.number_input(
    "Moisture Target (%)",
    min_value=0,
    max_value=100,
    value=45,
    step=1
)

blanch_data = df[
df["Ingredient Number"] == bean
]

import numpy as np

blanch_data = blanch_data.sort_values(
by="Percent"
)

blanch_data = (
    blanch_data
    .groupby("Blanch Time (Min)", as_index=False)
    ["Percent"]
    .mean()
)

blanch_data["Blanch Time (Min)"] = (
blanch_data["Blanch Time (Min)"].cummax()
)

blanch_data["Percent Rounded"] = blanch_data["Percent"].round(0)
target_rounded = round(moisture_target)

import math

min_percent = math.floor(
    blanch_data["Percent"].min()
)
max_percent = math.ceil(
    blanch_data["Percent"].max()
)
    
import numpy as np

from sklearn.metrics import r2_score

if st.button("Calculate Blanch Time"):
    ingredient_data = blanch_data

    if len(ingredient_data) < 4:
        st.warning(
        f"""
    Insufficient data for a reliable prediction.

    Only {len(ingredient_data)} data point(s) exist for this ingredient.
    Use the historical results below instead:
    """
         )

        for _, row in ingredient_data.iterrows():
            st.write(
                f"• {row['Blanch Time (Min)']:.1f} min → "
                f"{row['Percent']:.1f} % moisture"
            )
    else:
        if moisture_target < min_percent or moisture_target > max_percent:
            st.error(
            f"""
            Moisture target is outside previously collected data range.
            Available predictions for {bean}:
            {min_percent:.1f}% to {max_percent:.1f}%
            Please choose a target within this range or collect additional data.
            """
        )

        else:
            x = ingredient_data["Blanch Time (Min)"]
            y = ingredient_data["Percent"]

            a, b = np.polyfit(np.log(x), y, 1)

            predicted_time = np.exp((moisture_target - b) / a)

            st.success(
                f"Recommended blanch time: {predicted_time:.1f} minutes"
            )        

            y_pred = a * np.log(x) + b
            r_squared = r2_score(y, y_pred)

            if a <= 0:
                st.warning(
                    "Historical data does not support a physically realistic relationship."
            )

            if r_squared < 0.60:
                closest_points = blanch_data.iloc[
                (blanch_data["Percent"] - moisture_target).abs().argsort()[:2]
                ]

                p1 = closest_points.iloc[0]
                p2 = closest_points.iloc[1]

                st.warning(
            f"""
            R² = {r_squared:.2f}

            Warning: Data for this ingredient shows the logarithmic relationship is weak and therefore prediction should be taken with a grain of salt.
            For reference, the nearest data points:

            • {p1['Blanch Time (Min)']:.1f} min → {p1['Percent']:.1f}% moisture

            • {p2['Blanch Time (Min)']:.1f} min → {p2['Percent']:.1f}% moisture
            """
        )


with st.expander("Add New Blanch Data"):

    new_bean = st.text_input(
        "Ingredient Number",
        placeholder="e.g. BEAN0001"
    )

    new_time = st.number_input(
        "Blanch Time (Min)",
        min_value=0.0
    )

    new_moisture = st.number_input(
        "Moisture (%)",
        min_value=0.0,
        max_value=100.0
    )

    if st.button("Save Data"):
        new_row = pd.DataFrame({
        "Ingredient Number": [new_bean],
        "Blanch Time (Min)": [new_time],
        "Percent": [new_moisture]
        })

        updated_df = pd.concat(
            [df, new_row],
            ignore_index=True
        )

        updated_df.to_csv(
            "blanch_data.csv",
            index=False
        )

        st.success("Data saved!")