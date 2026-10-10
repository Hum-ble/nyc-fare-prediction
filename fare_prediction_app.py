
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import date, time
from io import BytesIO

# Page setup
st.set_page_config(
    page_title="Ride-Hailing Fare Decision Support",
    layout="wide"
)

# Limit page width
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("lgb_final_model_7m.joblib")

# Load Taxi Zone Lookup
@st.cache_data
def load_zones():
    return pd.read_csv("taxi_zone_lookup_streamlit_final.csv")

model = load_model()
taxi_zones = load_zones()

# Fitted model components
preprocessor = model.named_steps["preprocessor"]
lgb_model = model.named_steps["model"]
feature_names = preprocessor.get_feature_names_out()

# SHAP explainer
@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(lgb_model)

explainer = load_explainer()

# SHAP predictor groups
shap_groups = {
    "request_month": "onehot__request_month_",
    "request_dayofweek": "onehot__request_dayofweek_",
    "request_hour": "onehot__request_hour_",
    "platform": "onehot__platform_",
    "shared_request_flag": "onehot__shared_request_flag_",
    "wav_request_flag": "onehot__wav_request_flag_",
    "access_a_ride_flag": "onehot__access_a_ride_flag_",
    "PULocationID": "target__PULocationID",
    "DOLocationID": "target__DOLocationID",
    "trip_miles": "remainder__trip_miles",
    "trip_time": "remainder__trip_time"
}

# User-friendly names
display_names = {
    "request_month": "Request month",
    "request_dayofweek": "Request day",
    "request_hour": "Request hour",
    "platform": "Platform",
    "PULocationID": "Pickup location",
    "DOLocationID": "Drop-off location",
    "shared_request_flag": "Shared ride request",
    "wav_request_flag": "WAV request",
    "access_a_ride_flag": "Access-A-Ride",
    "trip_miles": "Trip miles",
    "trip_time": "Trip time"
}

# Start on Page 1
if "page" not in st.session_state:
    st.session_state.page = "input"


# PAGE 1: TRIP INPUT
if st.session_state.page == "input":

    st.title("Ride-Hailing Passenger Fare Decision Support")
    st.caption("Page 1 of 2: Trip Input")

    st.write(
        "Enter the trip details below to estimate the base passenger fare "
        "and generate an explanation of the prediction."
    )

    # Trip information
    st.subheader("Trip Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        request_date = st.date_input(
            "Request date",
            value=date(2026, 4, 10),
            min_value=date(2026, 1, 1),
            max_value=date(2026, 5, 31),
            key="request_date"
        )

    with col2:
        request_time = st.time_input(
            "Request time",
            value=time(14, 0),
            key="request_time"
        )

    with col3:
        platform = st.selectbox(
            "Platform",
            ["Uber", "Lyft"],
            key="platform"
        )

    request_datetime = pd.Timestamp.combine(
        request_date,
        request_time
    )

    request_month = request_datetime.month_name()
    request_dayofweek = request_datetime.day_name()
    request_hour = request_datetime.hour

    # Pickup and drop-off
    st.subheader("Pickup and Drop-off Location")

    pickup_boroughs = sorted(
        taxi_zones.loc[
            taxi_zones["pickup_supported"],
            "Borough"
        ].unique()
    )

    dropoff_boroughs = sorted(
        taxi_zones.loc[
            taxi_zones["dropoff_supported"]
            & (taxi_zones["Borough"] != "Unknown"),
            "Borough"
        ].unique()
    )

    col4, col5 = st.columns(2)

    with col4:
        pickup_borough = st.selectbox(
            "Pickup borough",
            pickup_boroughs,
            key="pickup_borough"
        )

        pickup_zone_table = taxi_zones[
            (taxi_zones["Borough"] == pickup_borough)
            & (taxi_zones["pickup_supported"])
        ].sort_values("zone_display")

        pickup_options = pickup_zone_table[
            "zone_display"
        ].tolist()

        if st.session_state.get("pickup_zone") not in pickup_options:
            st.session_state.pickup_zone = pickup_options[0]

        pickup_zone = st.selectbox(
            "Pickup zone",
            pickup_options,
            key="pickup_zone"
        )

    with col5:
        dropoff_borough = st.selectbox(
            "Drop-off borough",
            dropoff_boroughs,
            key="dropoff_borough"
        )

        dropoff_zone_table = taxi_zones[
            (taxi_zones["Borough"] == dropoff_borough)
            & (taxi_zones["dropoff_supported"])
        ].sort_values("zone_display")

        dropoff_options = dropoff_zone_table[
            "zone_display"
        ].tolist()

        if st.session_state.get("dropoff_zone") not in dropoff_options:
            st.session_state.dropoff_zone = dropoff_options[0]

        dropoff_zone = st.selectbox(
            "Drop-off zone",
            dropoff_options,
            key="dropoff_zone"
        )

    pickup_location_id = int(
        pickup_zone_table.loc[
            pickup_zone_table["zone_display"] == pickup_zone,
            "LocationID"
        ].iloc[0]
    )

    dropoff_location_id = int(
        dropoff_zone_table.loc[
            dropoff_zone_table["zone_display"] == dropoff_zone,
            "LocationID"
        ].iloc[0]
    )

    # Trip details
    st.subheader("Trip Details")

    col6, col7, col8 = st.columns(3)

    with col6:
        trip_miles = st.number_input(
            "Trip distance (miles)",
            min_value=0.01,
            value=3.50,
            step=0.10,
            key="trip_miles"
        )

    with col7:
        trip_minutes = st.number_input(
            "Trip duration (minutes)",
            min_value=0,
            value=15,
            step=1,
            key="trip_minutes"
        )

    with col8:
        trip_seconds = st.number_input(
            "Additional seconds",
            min_value=0,
            max_value=59,
            value=30,
            step=1,
            key="trip_seconds"
        )

    trip_time = int(
        (trip_minutes * 60)
        + trip_seconds
    )

    # Service requests
    st.subheader("Service Requests")

    col9, col10, col11 = st.columns(3)

    with col9:
        shared_request = st.selectbox(
            "Shared ride requested?",
            ["No", "Yes"],
            key="shared_request"
        )

    with col10:
        wav_request = st.selectbox(
            "Wheelchair-accessible vehicle requested?",
            ["No", "Yes"],
            key="wav_request"
        )

    with col11:
        access_a_ride = st.selectbox(
            "Access-A-Ride?",
            ["No", "Yes"],
            key="access_a_ride"
        )

    shared_request_flag = (
        "Y" if shared_request == "Yes" else "N"
    )

    wav_request_flag = (
        "Y" if wav_request == "Yes" else "N"
    )

    access_a_ride_flag = (
        "Y" if access_a_ride == "Yes" else "N"
    )

    # Prediction
    if st.button("Predict Fare", type="primary"):

        if trip_time == 0:
            st.error(
                "Trip duration must be greater than zero."
            )

        else:
            trip_input = pd.DataFrame([{
                "request_month": request_month,
                "request_dayofweek": request_dayofweek,
                "request_hour": request_hour,
                "platform": platform,
                "PULocationID": pickup_location_id,
                "DOLocationID": dropoff_location_id,
                "shared_request_flag": shared_request_flag,
                "wav_request_flag": wav_request_flag,
                "access_a_ride_flag": access_a_ride_flag,
                "trip_miles": trip_miles,
                "trip_time": trip_time
            }])

            predicted_fare = model.predict(
                trip_input
            )[0]

            st.session_state.trip_input = trip_input
            st.session_state.predicted_fare = predicted_fare

            st.session_state.trip_summary = {
                "date": request_date,
                "time": request_time,
                "platform": platform,
                "pickup": f"{pickup_zone}, {pickup_borough}",
                "dropoff": f"{dropoff_zone}, {dropoff_borough}",
                "trip_miles": trip_miles,
                "trip_minutes": trip_minutes,
                "trip_seconds": trip_seconds,
                "shared_request": shared_request,
                "wav_request": wav_request,
                "access_a_ride": access_a_ride
            }

            st.session_state.page = "result"
            st.rerun()


# PAGE 2: PREDICTION AND EXPLANATION
else:

    trip_input = st.session_state.trip_input
    predicted_fare = st.session_state.predicted_fare
    summary = st.session_state.trip_summary

    st.title("Passenger Fare Prediction")
    st.caption(
        "Page 2 of 2: Prediction and Explanation"
    )

    # Predicted fare
    st.subheader("Predicted Passenger Fare")

    st.metric(
        label="Estimated base passenger fare",
        value=f"USD {predicted_fare:.2f}"
    )

    # Trip Summary: responsive mobile and desktop layout
    st.subheader("Trip Summary")

    # Format the existing prediction inputs for display
    from html import escape

    def summary_field(label, value):
        return (
            '<div class="fare-summary-field">'
            f'<div class="fare-summary-label">{escape(str(label))}</div>'
            f'<div class="fare-summary-value">{escape(str(value))}</div>'
            '</div>'
        )

    summary_date = summary["date"].strftime("%d %b %Y")
    summary_time = summary["time"].strftime("%H:%M")
    summary_distance = f'{summary["trip_miles"]:.2f} miles'
    summary_duration = (
        f'{summary["trip_minutes"]} min '
        f'{summary["trip_seconds"]} sec'
    )

    summary_css = """
    <style>
        .fare-summary {
            width: 100%;
            font-family: inherit;
            color: inherit;
        }

        .fare-summary-heading {
            margin: 0.3rem 0 1rem;
            text-align: center;
            font-size: 1.05rem;
            font-weight: 650;
        }

        .fare-summary-heading:not(:first-child) {
            margin-top: 1.5rem;
        }

        .fare-summary-grid {
            display: grid;
            gap: 1rem 1.5rem;
        }

        .fare-summary-info,
        .fare-summary-services {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .fare-summary-route,
        .fare-summary-details {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .fare-summary-field {
            min-width: 0;
        }

        .fare-summary-label {
            color: #8f8f98;
            font-size: 0.88rem;
            margin-bottom: 0.35rem;
        }

        .fare-summary-value {
            font-size: 1rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        /* Preserve the desktop column alignment */
        .fare-summary-info .fare-summary-field:nth-child(2),
        .fare-summary-services .fare-summary-field:nth-child(2) {
            text-align: center;
        }

        .fare-summary-info .fare-summary-field:last-child,
        .fare-summary-services .fare-summary-field:last-child,
        .fare-summary-route .fare-summary-field:last-child,
        .fare-summary-details .fare-summary-field:last-child {
            text-align: right;
        }

        /* Compact layout for phones */
        @media (max-width: 640px) {
            .fare-summary-grid {
                grid-template-columns: 1fr;
                gap: 0;
            }

            .fare-summary-field {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                gap: 1rem;
                padding: 0.65rem 0;
                border-bottom: 1px solid rgba(128, 128, 128, 0.20);
                text-align: left !important;
            }

            .fare-summary-label {
                margin-bottom: 0;
                flex-shrink: 0;
            }

            .fare-summary-value {
                text-align: right;
                font-size: 0.95rem;
            }

            .fare-summary-route .fare-summary-field {
                display: block;
            }

            .fare-summary-route .fare-summary-value {
                text-align: left;
                margin-top: 0.3rem;
            }

            .fare-summary-heading {
                margin: 0.3rem 0 0.6rem;
            }

            .fare-summary-heading:not(:first-child) {
                margin-top: 1.1rem;
            }
        }
    </style>
    """

    summary_html = f"""
    <div class="fare-summary">
        <div class="fare-summary-heading">Trip Information</div>
        <div class="fare-summary-grid fare-summary-info">
            {summary_field("Platform", summary["platform"])}
            {summary_field("Date", summary_date)}
            {summary_field("Time", summary_time)}
        </div>

        <div class="fare-summary-heading">Route</div>
        <div class="fare-summary-grid fare-summary-route">
            {summary_field("Pickup", summary["pickup"])}
            {summary_field("Drop-off", summary["dropoff"])}
        </div>

        <div class="fare-summary-grid fare-summary-details"
             style="margin-top: 1rem;">
            {summary_field("Distance", summary_distance)}
            {summary_field("Duration", summary_duration)}
        </div>

        <div class="fare-summary-heading">Service Requests</div>
        <div class="fare-summary-grid fare-summary-services">
            {summary_field("Shared ride", summary["shared_request"])}
            {summary_field("WAV request", summary["wav_request"])}
            {summary_field("Access-A-Ride", summary["access_a_ride"])}
        </div>
    </div>
    """

    with st.container(border=True):
        st.html(summary_css + summary_html)

    # SHAP explanation
    st.subheader(
        "Why did the model predict this fare?"
    )

    trip_transformed = preprocessor.transform(
        trip_input
    )

    trip_shap_values = explainer(
        trip_transformed
    )

    if hasattr(
        trip_shap_values.values,
        "toarray"
    ):
        shap_array = (
            trip_shap_values.values.toarray()
        )
    else:
        shap_array = np.asarray(
            trip_shap_values.values
        )

    grouped_values = {}

    for predictor, prefix in shap_groups.items():

        idx = [
            i
            for i, name in enumerate(feature_names)
            if name.startswith(prefix)
        ]

        grouped_values[predictor] = (
            shap_array[0, idx].sum()
        )

    grouped_shap = pd.Series(
        grouped_values
    )

    local_shap = shap.Explanation(
        values=grouped_shap.values,
        base_values=trip_shap_values.base_values[0],
        feature_names=[
            display_names[name]
            for name in grouped_shap.index
        ]
    )

    # Create SHAP waterfall
    shap.plots.waterfall(
        local_shap,
        max_display=11,
        show=False
    )

    fig = plt.gcf()
    fig.set_size_inches(8, 5)

    # Save high-resolution PNG
    plot_buffer = BytesIO()

    fig.savefig(
        plot_buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white"
    )

    plot_buffer.seek(0)
    plot_bytes = plot_buffer.getvalue()

    # Display compact chart
    st.image(
        plot_bytes,
        width= 900
    )

    plt.close(fig)

    st.caption(
        "Red contributions increase the predicted fare, "
        "while blue contributions reduce it relative "
        "to the SHAP baseline."
    )

    # Download SHAP chart
    st.download_button(
        label="Download SHAP Explanation",
        data=plot_bytes,
        file_name="fare_prediction_shap_explanation.png",
        mime="image/png"
    )

    # Predictor contribution values
    input_values = {
        "request_month":
            trip_input.iloc[0]["request_month"],

        "request_dayofweek":
            trip_input.iloc[0]["request_dayofweek"],

        "request_hour":
            f"{int(trip_input.iloc[0]['request_hour'])}:00",

        "platform":
            summary["platform"],

        "PULocationID":
            summary["pickup"],

        "DOLocationID":
            summary["dropoff"],

        "shared_request_flag":
            summary["shared_request"],

        "wav_request_flag":
            summary["wav_request"],

        "access_a_ride_flag":
            summary["access_a_ride"],

        "trip_miles":
            f"{summary['trip_miles']:.2f}",

        "trip_time":
            (
                f"{summary['trip_minutes']} min "
                f"{summary['trip_seconds']} sec"
            )
    }

    contribution_table = pd.DataFrame({
        "Predictor": [
            display_names[name]
            for name in grouped_shap.index
        ],
        "Input": [
            input_values[name]
            for name in grouped_shap.index
        ],
        "SHAP contribution (USD)":
            grouped_shap.values
    })

    contribution_table[
        "Absolute contribution"
    ] = (
        contribution_table[
            "SHAP contribution (USD)"
        ].abs()
    )

    contribution_table = (
        contribution_table
        .sort_values(
            "Absolute contribution",
            ascending=False
        )
        .drop(
            columns="Absolute contribution"
        )
        .reset_index(drop=True)
    )

    contribution_table[
        "SHAP contribution (USD)"
    ] = (
        contribution_table[
            "SHAP contribution (USD)"
        ].round(2)
    )

    # Predictor contribution table
    st.subheader("Predictor Contributions")

    st.dataframe(
        contribution_table,
        use_container_width=True,
        hide_index=True,
        height=460
    )

    # Return to Page 1
    if st.button("Modify Trip Inputs"):
        st.session_state.page = "input"
        st.rerun()
