"""
Generates the workshop dataset for Notebook 1: Day-Ahead Solar Forecasting.

Data source : Open-Meteo Historical API
Location    : Powerhouse Energy Campus, CSU, Fort Collins, CO  (40.5944 N, 105.07594 W)
Period      : June 1 2025 – June 1 2026  (one full year, hourly)
Output      : data/workshop/solar_forecasting.csv

Features in the output CSV
---------------------------
Calendar (always known in advance)
  hour              : 0–23
  day_of_year       : 1–365
  sin_hour          : sine encoding of hour
  cos_hour          : cosine encoding of hour
  sin_doy           : sine encoding of day_of_year
  cos_doy           : cosine encoding of day_of_year

Weather (perfect-forecast proxy — tomorrow's actual observed values)
  temp_forecast     : temperature_2m  shifted -24 h  (°C)
  cloud_forecast    : cloud_cover     shifted -24 h  (%)
  humidity_forecast : relative_humidity_2m shifted -24 h  (%)
  precip_forecast   : precipitation   shifted -24 h  (mm)

Physics
  clearsky_ghi      : theoretical clear-sky GHI from pvlib (W/m²)

Lag features (information available before the forecast is made)
  solar_rad_t24     : shortwave_radiation shifted +24 h  — same hour yesterday
  solar_rad_t48     : shortwave_radiation shifted +48 h  — same hour two days ago

Target
  solar_radiation   : shortwave_radiation at time t  (W/m²)
  target_t24        : shortwave_radiation at time t+24 — what we are predicting

"""

import sys
import time
import requests
import numpy as np
import pandas as pd
import pvlib
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

LATITUDE   = 40.594424
LONGITUDE  = -105.075948
ALTITUDE   = 1525          # Fort Collins elevation in metres
TIMEZONE   = "America/Denver"
START_DATE = "2025-06-01"
END_DATE   = "2026-06-01"

OUTPUT_DIR  = Path(__file__).resolve().parent.parent / "data" / "workshop"
OUTPUT_FILE = OUTPUT_DIR / "solar_forecasting.csv"

RANDOM_SEED = 469           # Matches seeds.py convention

# ── Step 1: Fetch data from Open-Meteo Historical API ────────────────────────

def fetch_open_meteo(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """
    Fetch hourly weather and solar radiation data from Open-Meteo archive.

    Returns a DataFrame indexed by timezone-aware timestamp.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : latitude,
        "longitude" : longitude,
        "start_date": start_date,
        "end_date"  : end_date,
        "hourly"    : ",".join([
            "shortwave_radiation",
            "temperature_2m",
            "cloud_cover",
            "relative_humidity_2m",
            "precipitation",
        ]),
        "timezone"  : "UTC",  # Request UTC to avoid DST ambiguity
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"  Fetching Open-Meteo data (attempt {attempt}/{retries})...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            print(f"  Warning: attempt {attempt} failed — {exc}")
            if attempt == retries:
                raise RuntimeError(
                    "Could not fetch Open-Meteo data after "
                    f"{retries} attempts."
                ) from exc
            time.sleep(backoff * attempt)

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)

    # Parse timestamps as UTC — avoids DST ambiguity entirely.
    # Open-Meteo returns clean UTC when timezone="UTC" is requested.
    # Local hour-of-day is computed in engineer_features via UTC offset.
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"]).set_index("timestamp")

    print(f"  Fetched {len(df):,} hourly rows "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── Step 2: Compute clear-sky GHI via pvlib ──────────────────────────────────

def compute_clearsky(df: pd.DataFrame, latitude: float, longitude: float,
                     altitude: float) -> pd.Series:
    """
    Compute theoretical clear-sky GHI using the Ineichen model in pvlib.

    This is a physics-based feature: the maximum possible solar radiation
    given the sun's position and atmospheric conditions, with no clouds.
    It gives the model a strong prior about the diurnal and seasonal cycle.
    """
    location = pvlib.location.Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        tz=df.index.tz,
    )

    # Compute solar position and clear-sky irradiance
    clearsky = location.get_clearsky(df.index, model="ineichen")

    # ghi = global horizontal irradiance (W/m²)
    return clearsky["ghi"].rename("clearsky_ghi")


# ── Step 3: Feature engineering ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct all features needed for day-ahead solar forecasting.

    Each row represents one hour.
    The model predicts solar_radiation at t+24 (target_t24)
    using only information available at or before time t.
    """
    out = pd.DataFrame(index=df.index)

    # ── Calendar features ──────────────────────────────────────────────────
    # Sine/cosine encoding avoids the discontinuity at hour 23 → 0
    # and day 365 → 1, which would confuse distance-based models.
    # Convert UTC index to local Mountain Time for meaningful hour/doy.

    local_index = df.index.tz_convert("America/Denver")
    hour      = local_index.hour.astype(float)
    doy       = local_index.dayofyear.astype(float)

    out["hour"]       = hour.astype(int)
    out["day_of_year"]= doy.astype(int)
    out["sin_hour"]   = np.sin(2 * np.pi * hour / 24)
    out["cos_hour"]   = np.cos(2 * np.pi * hour / 24)
    out["sin_doy"]    = np.sin(2 * np.pi * doy / 365)
    out["cos_doy"]    = np.cos(2 * np.pi * doy / 365)

    # ── Weather forecast features (perfect-forecast proxy) ─────────────────
    # In production these come from a weather forecast API (e.g. Open-Meteo
    # forecast endpoint or NOAA GFS). Here we use tomorrow's actual observed
    # values — a standard academic benchmark called "oracle" or "perfect
    # forecast" inputs. This isolates the solar forecasting problem from
    # the weather forecasting problem.
    #
    # shift(-24) moves tomorrow's value into today's row.
    # NaN rows at the end of the series are dropped later.

    out["temp_forecast"]     = df["temperature_2m"].shift(-24)
    out["cloud_forecast"]    = df["cloud_cover"].shift(-24)
    out["humidity_forecast"] = df["relative_humidity_2m"].shift(-24)
    out["precip_forecast"]   = df["precipitation"].shift(-24)

    # ── Physics feature ────────────────────────────────────────────────────
    out["clearsky_ghi"] = df["clearsky_ghi"]

    # ── Lag features ───────────────────────────────────────────────────────
    # shift(+24) moves yesterday's same-hour value into today's row.
    # These are strictly causal: both are observed before the forecast is made.

    out["solar_rad_t24"] = df["shortwave_radiation"].shift(24)   # yesterday
    out["solar_rad_t48"] = df["shortwave_radiation"].shift(48)   # two days ago

    # ── Current observation and target ────────────────────────────────────
    out["solar_radiation"] = df["shortwave_radiation"]

    # target_t24: what we want to predict — same-hour value tomorrow
    # shift(-24) moves tomorrow's observation into today's row.
    out["target_t24"] = df["shortwave_radiation"].shift(-24)

    return out


# ── Step 4: Clean and validate ────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN (from lag/shift boundaries) and clip negatives.
    """
    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)

    print(f"  Dropped {n_before - n_after} rows with NaN "
          f"(lag/forecast boundaries)")

    # Solar radiation cannot be negative — clip any floating-point noise
    for col in ["solar_radiation", "solar_rad_t24", "solar_rad_t48",
                "clearsky_ghi", "target_t24"]:
        df[col] = df[col].clip(lower=0.0)

    return df


def validate(df: pd.DataFrame) -> None:
    """Basic sanity checks before saving."""
    assert df.isnull().sum().sum() == 0, "NaN values remain after cleaning"
    assert (df["target_t24"] >= 0).all(), "Negative target values"
    assert (df["clearsky_ghi"] >= 0).all(), "Negative clear-sky GHI"
    assert len(df) > 5000, f"Too few rows after cleaning: {len(df)}"
    print(f"  Validation passed — {len(df):,} rows, "
          f"{df.columns.tolist()}")


# ── Step 5: Train / val / test split summary ─────────────────────────────────

def print_split_summary(df: pd.DataFrame) -> None:
    """
    Print the time-ordered split that the notebook will use.
    (Splitting is done inside the notebook; this is just informational.)

    Train: first 70%  — model sees this during training
    Val  : next  15%  — used for early stopping / hyperparameter tuning
    Test : last  15%  — held out, touched only for final evaluation
    """
    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val

    train_end = df.index[n_train - 1]
    val_end   = df.index[n_train + n_val - 1]
    test_end  = df.index[-1]

    print("\n  Recommended time-ordered split:")
    print(f"    Train : {df.index[0].date()} → {train_end.date()}"
          f"  ({n_train:,} rows, 70%)")
    print(f"    Val   : {df.index[n_train].date()} → {val_end.date()}"
          f"  ({n_val:,} rows, 15%)")
    print(f"    Test  : {df.index[n_train + n_val].date()} → {test_end.date()}"
          f"  ({n_test:,} rows, 15%)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Solar Forecasting Dataset Generator")
    print(f"  Location : Fort Collins, CO "
          f"({LATITUDE}° N, {LONGITUDE}° W)")
    print(f"  Period   : {START_DATE} → {END_DATE}")
    print("=" * 60)

    # 1. Fetch raw data
    print("\n[1/5] Fetching Open-Meteo historical data...")
    raw = fetch_open_meteo(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        start_date=START_DATE,
        end_date=END_DATE,
        timezone=TIMEZONE,
    )

    # 2. Clear-sky GHI
    print("\n[2/5] Computing clear-sky GHI via pvlib (Ineichen model)...")
    raw["clearsky_ghi"] = compute_clearsky(
        raw, LATITUDE, LONGITUDE, ALTITUDE
    )
    print(f"  Clear-sky GHI range: "
          f"{raw['clearsky_ghi'].min():.1f} – "
          f"{raw['clearsky_ghi'].max():.1f} W/m²")

    # 3. Feature engineering
    print("\n[3/5] Engineering features...")
    featured = engineer_features(raw)

    # 4. Clean
    print("\n[4/5] Cleaning dataset...")
    clean_df = clean(featured)
    validate(clean_df)

    # 5. Save
    print("\n[5/5] Saving dataset...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert index from UTC to Mountain Time for readability
    clean_df.index = clean_df.index.tz_convert("America/Denver")

    clean_df.to_csv(OUTPUT_FILE)

    # Summary
    print_split_summary(clean_df)

    print("\n" + "=" * 60)
    print("  Done. Load in your notebook with:")
    print("  df = pd.read_csv('data/workshop/solar_forecasting.csv',")
    print("                   index_col='timestamp', parse_dates=True)")
    print("=" * 60)


if __name__ == "__main__":
    main()