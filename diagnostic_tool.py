"""
Quick diagnostic script to check data structure and fix common issues
Run this if you encounter column errors
"""

import pandas as pd
import fastf1
import os

# Import your race configuration
from race_config_week24 import (
    YEAR, RACE_NUMBER, SESSION_TYPE,
    CLEAN_AIR_RACE_PACE, QUALIFYING_DATA, TEAM_POINTS,
    DRIVER_TO_TEAM, WEATHER_LAT, WEATHER_LON, FORECAST_TIME
)

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")


print("F1 PREDICTION MODEL - DIAGNOSTIC TOOL")

print()

# Check qualifying data structure
print("1. CHECKING QUALIFYING DATA STRUCTURE")
print("-"*80)
qualifying_df = pd.DataFrame(QUALIFYING_DATA)
print("Columns in QUALIFYING_DATA:")
print(qualifying_df.columns.tolist())
print("\nFirst few rows:")
print(qualifying_df.head())
print()

# Check if session loads correctly
print("2. CHECKING SESSION DATA")
print("-"*80)
try:
    session = fastf1.get_session(YEAR, RACE_NUMBER, SESSION_TYPE)
    session.load()
    print(f" Session loaded successfully: {session.event['EventName']} - {SESSION_TYPE}")
    print(f" Number of drivers: {len(session.laps['Driver'].unique())}")
    print(f" Drivers: {sorted(session.laps['Driver'].unique())}")
except Exception as e:
    print(f" Error loading session: {e}")
print()

# Check laps data structure
print("3. CHECKING LAPS DATA STRUCTURE")
print("-"*80)
if 'session' in locals():
    laps = session.laps
    print("Available columns in laps data:")
    print(laps.columns.tolist())
    print(f"\nTotal laps: {len(laps)}")
    print(f"Laps with valid LapTime: {laps['LapTime'].notna().sum()}")
print()

# Check driver matching
print("4. CHECKING DRIVER MATCHING")
print("-"*80)
if 'session' in locals():
    config_drivers = set(qualifying_df['Driver'].unique())
    session_drivers = set(session.laps['Driver'].unique())
    
    print(f"Drivers in config: {sorted(config_drivers)}")
    print(f"Drivers in session: {sorted(session_drivers)}")
    
    matching = config_drivers & session_drivers
    only_config = config_drivers - session_drivers
    only_session = session_drivers - config_drivers
    
    print(f"\n Matching drivers: {sorted(matching)}")
    if only_config:
        print(f" In config but not in session: {sorted(only_config)}")
    if only_session:
        print(f" In session but not in config: {sorted(only_session)}")
print()

# Check feature extraction
print("5. TESTING FEATURE EXTRACTION")
print("-"*80)
if 'session' in locals() and len(qualifying_df) > 0:
    try:
        # Test on first driver
        test_driver = qualifying_df.iloc[0]['Driver']
        print(f"Testing feature extraction for driver: {test_driver}")
        
        driver_laps = session.laps.pick_drivers(test_driver)
        print(f" Found {len(driver_laps)} laps for {test_driver}")
        
        if len(driver_laps) > 0:
            sector1_std = driver_laps['Sector1Time'].dt.total_seconds().std()
            print(f" Sector 1 consistency (std): {sector1_std:.3f}s")
            
            fastest_lap = driver_laps.pick_fastest()
            if fastest_lap is not None and not fastest_lap.empty:
                try:
                    telemetry = fastest_lap.get_telemetry()
                    if not telemetry.empty:
                        print(f" Telemetry data available: {len(telemetry)} samples")
                        print(f"  - Avg Speed: {telemetry['Speed'].mean():.1f} km/h")
                        print(f"  - Max Speed: {telemetry['Speed'].max():.1f} km/h")
                except Exception as e:
                    print(f" Could not get telemetry: {e}")
        
    except Exception as e:
        print(f" Error in feature extraction: {e}")
print()

# Provide recommendations
print("6. RECOMMENDATIONS")
print("-"*80)

issues_found = []

if 'only_config' in locals() and only_config:
    issues_found.append(
        f"Some drivers in your config are not in the session data: {only_config}\n"
        f"  -> Remove these drivers from QUALIFYING_DATA in race_config.py"
    )

if 'only_session' in locals() and only_session:
    issues_found.append(
        f"Some drivers from the session are missing in your config: {only_session}\n"
        f"  -> Add these drivers to QUALIFYING_DATA in race_config.py"
    )

# Check if qualifying data has the right columns
expected_cols = ['Driver', 'QualifyingTime (s)']
missing_cols = [col for col in expected_cols if col not in qualifying_df.columns]
if missing_cols:
    issues_found.append(
        f"Missing required columns in QUALIFYING_DATA: {missing_cols}\n"
        f"  -> Make sure each entry has 'Driver' and 'QualifyingTime (s)' keys"
    )

# Check CLEAN_AIR_RACE_PACE coverage
config_drivers = set(qualifying_df['Driver'].unique())
pace_drivers = set(CLEAN_AIR_RACE_PACE.keys())
missing_pace = config_drivers - pace_drivers
if missing_pace:
    issues_found.append(
        f"Missing clean air race pace data for: {missing_pace}\n"
        f"  -> Add these drivers to CLEAN_AIR_RACE_PACE in race_config.py"
    )

if issues_found:
    print(" ISSUES FOUND:")
    for i, issue in enumerate(issues_found, 1):
        print(f"\n{i}. {issue}")
else:
    print(" No major issues found! Your configuration looks good.")

print()

print("DIAGNOSTIC COMPLETE")

