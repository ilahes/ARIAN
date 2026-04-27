# arian-wildfire-prediction

## Problem Statement
Azerbaijan faces rising wildfire risk due to climate change — temperatures have increased +0.4 °C/decade since 1980 while summer precipitation declines. No regional 30-day fire-risk forecast exists. ARIAN integrates satellite, meteorological, and forest inventory data into a unified ML pipeline that delivers daily wildfire probability, expected fire counts, and 30-day weather forecasts for 16 Azerbaijani cities.

## Why It Matters
Wildfire incidents have increased measurably since 2010, with the 2021–2022 seasons causing substantial forest loss across the Greater and Lesser Caucasus (~1.2 M hectares). Early-warning forecasts give emergency services 7–14 days of lead time to pre-position resources, issue evacuations, and reduce human casualties and economic damage (~$70–80 M/year from extreme weather).

## Target
**`fire_occurred`** — binary (1 = ≥1 NASA FIRMS VIIRS hotspot within 50 km of city centroid on forecast date). Label window: next-day 00:00–23:59 local time.  
**`fire_count`** — expected number of FIRMS hotspots in the same buffer (non-negative integer; Poisson regression).

## Features

| Source | Name | Units | Aggregation |
|--------|------|-------|-------------|
| Open-Meteo | temperature_2m | °C | daily mean; lags 1/3/7/14 d |
| Open-Meteo | wind_speed_10m | m/s | daily mean; lags 1/3/7/14 d |
| Open-Meteo | relative_humidity_2m | % | daily mean; lags 1/3/7/14 d |
| Open-Meteo | precipitation_7d_sum | mm | rolling 7-day sum |
| Open-Meteo | rain_30d_sum | mm | rolling 30-day sum |
| Open-Meteo | heatwave_flag | binary | 3+ consecutive days above local p90 |
| FWI (computed) | fwi, ffmc, dc, dsr | — | daily value |
| SPEI (computed) | spei_3, spi_1 | σ | 3-month / 1-month standardized |
| ESA WorldCover | land_cover_class | class | dominant class in 50 km buffer |
| AZ Forest Boundary | forest_fraction | fraction | forest area / buffer area |
| MESE QURULUSU | dominant_species | class | modal species by stand area |
| MESE QURULUSU | mean_crown_density | 0.1–1.0 | area-weighted canopy closure |
| MESE QURULUSU | pct_old_growth | fraction | stands age ≥ 80 yr / total area |
| WorldPop + OSM | human_activity_score | — | population_density × road_density |
| NASA FIRMS | days_since_last_fire | days | days since last hotspot in buffer |
| NASA FIRMS | fire_count_30d | count | hotspot count in prior 30 days |

## Horizon
**t+1 to t+30 days** — daily wildfire risk scores for the next 30 days; 30-day multi-target weather forecasts (temperature, wind, precipitation).

## Dataset
| Source | Period | Granularity | Region |
|--------|--------|-------------|--------|
| Open-Meteo weather | 2020–present | Hourly + daily | 16 Azerbaijani cities |
| NASA FIRMS VIIRS-SNPP | 2020–2025 | ~375 m point events | Azerbaijan + 50 km city buffers |
| ESA WorldCover | 2020 | 10 m raster | Azerbaijan |
| WorldPop | 2020–2026 | 100 m annual raster | Azerbaijan |
| MESE QURULUSU Forest Inventory | Static | Stand-level polygons | National forest boundary |

## Key Definitions
- **FWI** — Canadian Fire Weather Index system (FFMC, DMC, DC, ISI, BUI, FWI, DSR); derived from temperature, humidity, wind, rain.
- **SPEI** — Standardized Precipitation-Evapotranspiration Index; drought measure that accounts for evaporative demand (unlike SPI).
- **FIRMS** — NASA Fire Information for Resource Management System; satellite fire hotspot detections from VIIRS-SNPP.
- **MESE QURULUSU** — Azerbaijani State Forest Management Inventory; stand-level species, crown density, and age class.
- **HGBC / HGBR** — HistGradientBoostingClassifier / Regressor (scikit-learn); primary wildfire models with isotonic calibration.
- **Mann-Kendall** — Non-parametric test for monotonic trend in climate time series.
- **AUC-ROC** — Area Under the ROC Curve; primary wildfire classifier evaluation metric (target ≥ 0.85).

## Team & Roles

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Raul | Data Collection & Model Development | Phase 1 — ingestion pipeline, data acquisition; Phase 4 — ML model training & evaluation |
| Aysu | Preprocessing & Web | Phase 2 — data cleaning, EDA; Phase 5 — FastAPI + dashboard |
| Ilaha | Preprocessing & Model Development | Phase 2 — data cleaning, EDA; Phase 4 — model experiments & evaluation |
| Asif | Feature Engineering, Modeling & Web | Phase 3 — FWI/SPEI/forest feature engineering; Phase 4 — modeling; Phase 5 — web |
| Nurana | Preprocessing & Web | Phase 2 — data cleaning, EDA; Phase 5 — web interface |

## Daily Activities
2026-04-20 — All Team — Project kick-off; repo structure set up, Open-Meteo API explored, 16 cities and target variables selected.
2026-04-21 — Raul — Data ingestion pipeline built; full Open-Meteo historical fetch (hourly + daily, 2020–present) for all 16 cities completed.
2026-04-22 — Raul — Database design finalized; DuckDB schema implemented, raw data loaded and validated.
2026-04-23 — Aysu / Ilaha / Nurana — Data cleaning pipeline applied; feature engineering (lags, rolling windows, heatwave flag) completed for weather data.
2026-04-24 — Raul — Pipeline automation complete; orchestrator (`generate_data.py`) with incremental loading, quality gates, and logging operational.
2026-04-25 — Asif — ARIAN v3.1 blueprint finalized; all Tier-0 datasets confirmed collected (Open-Meteo, FIRMS, ESA WorldCover, WorldPop, OSM, Azerbaycan.kmz, MESE QURULUSU forest inventory).
2026-04-27 — All Team — Summarized README created; EDA phase (Day 6) begins — descriptive statistics, distributions, time-series cross-city comparison.
