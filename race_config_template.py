"""
Race Configuration File - Template
Copy this file and update values for each race weekend
"""

# ==================== RACE INFORMATION ====================
YEAR = 2025
RACE_NUMBER = 6  # Update for each race (1-24)
SESSION_TYPE = 'R'  # 'R' for Race, 'Q' for Qualifying, 'FP1/FP2/FP3' for Practice

# ==================== QUALIFYING DATA ====================
# Update these with actual qualifying times in seconds
QUALIFYING_DATA = [
    {"Driver": "VER", "QualifyingTime (s)": 78.123},
    {"Driver": "PER", "QualifyingTime (s)": 78.456},
    {"Driver": "LEC", "QualifyingTime (s)": 78.234},
    {"Driver": "SAI", "QualifyingTime (s)": 78.345},
    {"Driver": "HAM", "QualifyingTime (s)": 78.567},
    {"Driver": "RUS", "QualifyingTime (s)": 78.678},
    {"Driver": "NOR", "QualifyingTime (s)": 78.789},
    {"Driver": "PIA", "QualifyingTime (s)": 78.890},
    {"Driver": "ALO", "QualifyingTime (s)": 79.012},
    {"Driver": "STR", "QualifyingTime (s)": 79.123},
    {"Driver": "GAS", "QualifyingTime (s)": 79.234},
    {"Driver": "OCO", "QualifyingTime (s)": 79.345},
    {"Driver": "TSU", "QualifyingTime (s)": 79.456},
    {"Driver": "RIC", "QualifyingTime (s)": 79.567},
    {"Driver": "BOT", "QualifyingTime (s)": 79.678},
    {"Driver": "ZHO", "QualifyingTime (s)": 79.789},
    {"Driver": "MAG", "QualifyingTime (s)": 79.890},
    {"Driver": "HUL", "QualifyingTime (s)": 79.901},
    {"Driver": "ALB", "QualifyingTime (s)": 80.012},
    {"Driver": "SAR", "QualifyingTime (s)": 80.123},
]

# ==================== CLEAN AIR RACE PACE ====================
# Average lap times from practice sessions in clean air (seconds)
# Estimate based on long runs during FP2/FP3
CLEAN_AIR_RACE_PACE = {
    "VER": 82.5,
    "PER": 82.8,
    "LEC": 82.6,
    "SAI": 82.9,
    "HAM": 83.0,
    "RUS": 83.1,
    "NOR": 83.2,
    "PIA": 83.4,
    "ALO": 83.5,
    "STR": 83.6,
    "GAS": 83.7,
    "OCO": 83.8,
    "TSU": 84.0,
    "RIC": 84.1,
    "BOT": 84.2,
    "ZHO": 84.3,
    "MAG": 84.4,
    "HUL": 84.5,
    "ALB": 84.6,
    "SAR": 84.7,
}

# ==================== TEAM PERFORMANCE ====================
# Current season constructor standings points
# Update after each race with actual points
TEAM_POINTS = {
    "Red Bull Racing": 123,
    "Ferrari": 98,
    "Mercedes": 87,
    "McLaren": 76,
    "Aston Martin": 54,
    "Alpine": 32,
    "RB": 28,
    "Kick Sauber": 12,
    "Haas F1 Team": 8,
    "Williams": 4,
}

# ==================== DRIVER TO TEAM MAPPING ====================
DRIVER_TO_TEAM = {
    "VER": "Red Bull Racing",
    "PER": "Red Bull Racing",
    "LEC": "Ferrari",
    "SAI": "Ferrari",
    "HAM": "Mercedes",
    "RUS": "Mercedes",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "OCO": "Alpine",
    "TSU": "RB",
    "RIC": "RB",
    "BOT": "Kick Sauber",
    "ZHO": "Kick Sauber",
    "MAG": "Haas F1 Team",
    "HUL": "Haas F1 Team",
    "ALB": "Williams",
    "SAR": "Williams",
}

# ==================== WEATHER LOCATION ====================
# Circuit GPS coordinates for weather forecast
# Examples:
# Monaco: 43.7347, 7.4206
# Silverstone: 52.0733, -1.0147
# Monza: 45.6156, 9.2811
# Spa: 50.4372, 5.9714
# Suzuka: 34.8431, 136.5408

WEATHER_LAT = 43.7347  # Update for circuit
WEATHER_LON = 7.4206   # Update for circuit

# ==================== RACE TIMING ====================
# Race start time in UTC format
# Format: "YYYY-MM-DD HH:MM:SS"
FORECAST_TIME = "2025-05-26 15:00:00"

# ==================== NOTES ====================
# How to fill this configuration:
#
# 1. QUALIFYING_DATA:
#    - After qualifying session, check official F1 timing
#    - Or use FastF1: session = fastf1.get_session(YEAR, RACE_NUMBER, 'Q')
#    - Convert lap times to seconds
#
# 2. CLEAN_AIR_RACE_PACE:
#    - Watch FP2 and FP3 long runs
#    - Note lap times when drivers are in clean air
#    - Average the times for each driver
#    - Or use FastF1 to analyze practice laps
#
# 3. TEAM_POINTS:
#    - Check official F1 standings after previous race
#    - https://www.formula1.com/en/results.html
#
# 4. DRIVER_TO_TEAM:
#    - Usually stays same unless mid-season changes
#    - Check official F1 entry list
#
# 5. WEATHER_LAT/LON:
#    - Google the circuit name + "coordinates"
#    - Or use: https://www.latlong.net/
#
# 6. FORECAST_TIME:
#    - Check F1 schedule for race start time
#    - Convert to UTC timezone
#    - Format: YYYY-MM-DD HH:MM:SS

# ==================== CIRCUIT-SPECIFIC DATA ====================
# You can add circuit-specific adjustments here
CIRCUIT_CHARACTERISTICS = {
    "name": "Circuit Name",
    "country": "Country",
    "lap_length_km": 0.0,  # Track length
    "corners": 0,  # Number of corners
    "drs_zones": 0,  # Number of DRS zones
    "overtaking_difficulty": "Medium",  # Easy/Medium/Hard
    "tire_wear": "Medium",  # Low/Medium/High
}

# ==================== STRATEGY CONSIDERATIONS ====================
# Optional: Add expected strategy information
EXPECTED_STRATEGY = {
    "number_of_stops": 1,  # Expected pit stops
    "dominant_compound": "MEDIUM",  # Expected main tire compound
    "safety_car_probability": 0.3,  # 0.0 to 1.0
}
