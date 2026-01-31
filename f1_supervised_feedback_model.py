import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import json
import pickle
from collections import deque

# Import race configuration
from race_config_week24 import (
    YEAR, RACE_NUMBER, SESSION_TYPE,
    CLEAN_AIR_RACE_PACE, QUALIFYING_DATA, TEAM_POINTS,
    DRIVER_TO_TEAM, WEATHER_LAT, WEATHER_LON, FORECAST_TIME
)

# Load environment variables
load_dotenv()

os.makedirs("f1_cache", exist_ok=True)
os.makedirs("model_checkpoints", exist_ok=True)
os.makedirs("feedback_data", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

# ==================== FEEDBACK SYSTEM ====================

class FeedbackManager:
    """Manages user feedback and model retraining based on actual race results"""
    
    def __init__(self, feedback_file="feedback_data/race_feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_history = self.load_feedback()
        
    def load_feedback(self):
        """Load historical feedback from file"""
        if os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return {"races": [], "corrections": []}
    
    def save_feedback(self):
        """Save feedback to file"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def add_race_result(self, race_id, predictions, actual_results, metadata=None):
        """Add actual race results for comparison with predictions"""
        feedback_entry = {
            "race_id": race_id,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "actual_results": actual_results,
            "metadata": metadata or {},
            "errors": self.calculate_errors(predictions, actual_results)
        }
        self.feedback_history["races"].append(feedback_entry)
        self.save_feedback()
        return feedback_entry
    
    def add_correction(self, driver, feature, old_value, new_value, reason):
        """Add manual correction based on expert knowledge"""
        correction = {
            "timestamp": datetime.now().isoformat(),
            "driver": driver,
            "feature": feature,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason
        }
        self.feedback_history["corrections"].append(correction)
        self.save_feedback()
    
    def calculate_errors(self, predictions, actual_results):
        """Calculate prediction errors"""
        pred_dict = {p["driver"]: p for p in predictions}
        actual_dict = {a["driver"]: a for a in actual_results}
        
        errors = []
        for driver in pred_dict.keys():
            if driver in actual_dict:
                pred_pos = pred_dict[driver].get("position", None)
                actual_pos = actual_dict[driver].get("position", None)
                pred_time = pred_dict[driver].get("time", None)
                actual_time = actual_dict[driver].get("time", None)
                
                error_entry = {"driver": driver}
                if pred_pos and actual_pos:
                    error_entry["position_error"] = abs(pred_pos - actual_pos)
                if pred_time and actual_time:
                    error_entry["time_error"] = abs(pred_time - actual_time)
                errors.append(error_entry)
        
        return errors
    
    def get_performance_trends(self, driver=None):
        """Get performance trends over time"""
        if not self.feedback_history["races"]:
            return None
        
        trends = {}
        for race in self.feedback_history["races"]:
            for error in race["errors"]:
                driver_name = error["driver"]
                if driver and driver_name != driver:
                    continue
                    
                if driver_name not in trends:
                    trends[driver_name] = {
                        "position_errors": [],
                        "time_errors": []
                    }
                
                if "position_error" in error:
                    trends[driver_name]["position_errors"].append(error["position_error"])
                if "time_error" in error:
                    trends[driver_name]["time_errors"].append(error["time_error"])
        
        return trends


# ==================== ENHANCED DATA PREPARATION ====================

def load_historical_data(years_back=3):
    """Load historical race data from FastF1 API for better training"""
    all_laps = []
    all_results = []
    
    current_year = YEAR
    
    for year_offset in range(years_back):
        year = current_year - year_offset
        try:
            print(f"Loading data from {year}...")
            
            # Get the race schedule for the year
            schedule = fastf1.get_event_schedule(year)
            
            # Load data for the same circuit if available
            for idx, event in schedule.iterrows():
                if event['EventFormat'] == 'conventional':
                    try:
                        session = fastf1.get_session(year, event['RoundNumber'], 'R')
                        session.load(telemetry=False, weather=False, messages=False)
                        
                        laps = session.laps[["Driver", "LapTime", "Sector1Time", 
                                             "Sector2Time", "Sector3Time", "Compound",
                                             "TyreLife", "TrackStatus"]].copy()
                        laps['Year'] = year
                        laps['RoundNumber'] = event['RoundNumber']
                        laps['EventName'] = event['EventName']
                        
                        all_laps.append(laps)
                        
                        # Get race results
                        results = session.results[['DriverNumber', 'Abbreviation', 'Position', 
                                                   'ClassifiedPosition', 'GridPosition', 
                                                   'Status', 'Points']].copy()
                        results['Year'] = year
                        results['RoundNumber'] = event['RoundNumber']
                        all_results.append(results)
                        
                    except Exception as e:
                        print(f"  Could not load {event['EventName']} {year}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Could not load schedule for {year}: {e}")
    
    if all_laps:
        historical_laps = pd.concat(all_laps, ignore_index=True)
        historical_results = pd.concat(all_results, ignore_index=True)
        return historical_laps, historical_results
    
    return None, None


def prepare_enhanced_features(session, qualifying_data, weather_data):
    """Prepare enhanced features using FastF1 API data"""
    
    # Get telemetry and timing data
    features = []
    
    for _, driver_qual in qualifying_data.iterrows():
        driver = driver_qual['Driver']
        
        # Get driver's laps from session (using pick_drivers to avoid deprecation warning)
        driver_laps = session.laps.pick_drivers(driver)
        
        if len(driver_laps) == 0:
            continue
        
        # Calculate sector consistency (lower is better)
        sector1_std = driver_laps['Sector1Time'].dt.total_seconds().std()
        sector2_std = driver_laps['Sector2Time'].dt.total_seconds().std()
        sector3_std = driver_laps['Sector3Time'].dt.total_seconds().std()
        
        # Get average speeds in each sector
        fastest_lap = driver_laps.pick_fastest()
        if fastest_lap is not None and not fastest_lap.empty:
            try:
                telemetry = fastest_lap.get_telemetry()
                avg_speed = telemetry['Speed'].mean() if not telemetry.empty else 0
                max_speed = telemetry['Speed'].max() if not telemetry.empty else 0
                avg_throttle = telemetry['Throttle'].mean() if not telemetry.empty else 0
            except:
                avg_speed = 0
                max_speed = 0
                avg_throttle = 0
        else:
            avg_speed = 0
            max_speed = 0
            avg_throttle = 0
        
        # Tire compound analysis
        tire_compounds = driver_laps['Compound'].value_counts()
        dominant_compound = tire_compounds.index[0] if len(tire_compounds) > 0 else 'UNKNOWN'
        
        # Track status (clean laps vs yellow flags)
        clean_laps = (driver_laps['TrackStatus'] == '1').sum()
        total_laps = len(driver_laps)
        clean_lap_ratio = clean_laps / total_laps if total_laps > 0 else 0
        
        feature_dict = {
            'Driver': driver,
            'Sector1Consistency': sector1_std if not pd.isna(sector1_std) else 0,
            'Sector2Consistency': sector2_std if not pd.isna(sector2_std) else 0,
            'Sector3Consistency': sector3_std if not pd.isna(sector3_std) else 0,
            'AvgSpeed': avg_speed,
            'MaxSpeed': max_speed,
            'AvgThrottle': avg_throttle,
            'CleanLapRatio': clean_lap_ratio,
            'DominantCompound': dominant_compound,
            'TotalLaps': total_laps
        }
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)


# ==================== ENHANCED DEEP LEARNING MODEL ====================

class F1Dataset(Dataset):
    """Custom PyTorch Dataset for F1 race predictions"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class F1RacePredictorWithUncertainty(nn.Module):
    """Enhanced Neural Network with uncertainty estimation"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.3):
        super(F1RacePredictorWithUncertainty, self).__init__()
        
        # Main prediction network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            # Use LayerNorm instead of BatchNorm to avoid issues with small batches
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer predicts both mean and variance (uncertainty)
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_size, 1)
        self.var_layer = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Softplus()  # Ensures positive variance
        )
    
    def forward(self, x):
        features = self.network(x)
        mean = self.mean_layer(features)
        variance = self.var_layer(features)
        return mean, variance
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Make predictions with uncertainty estimation using Monte Carlo dropout"""
        self.train()  # Keep dropout active
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                mean, _ = self.forward(x)
                predictions.append(mean)
        
        predictions = torch.stack(predictions)
        pred_mean = predictions.mean(dim=0)
        pred_std = predictions.std(dim=0)
        
        return pred_mean, pred_std


class OnlineLearningWrapper:
    """Wrapper for online learning with experience replay"""
    
    def __init__(self, model, optimizer, criterion, buffer_size=1000):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.experience_buffer = deque(maxlen=buffer_size)
        self.update_count = 0
    
    def add_experience(self, X, y):
        """Add new training example to experience buffer"""
        self.experience_buffer.append((X, y))
    
    def update_from_feedback(self, X_new, y_new, epochs=10, batch_size=4):
        """Update model with new feedback data"""
        # Add to experience buffer
        for x, y in zip(X_new, y_new):
            self.add_experience(x, y)
        
        # Sample from experience buffer for training
        if len(self.experience_buffer) < batch_size:
            return
        
        device = next(self.model.parameters()).device
        
        for epoch in range(epochs):
            # Sample batch from experience buffer
            indices = np.random.choice(len(self.experience_buffer), 
                                      size=min(batch_size, len(self.experience_buffer)), 
                                      replace=False)
            
            batch_X = torch.stack([self.experience_buffer[i][0] for i in indices]).to(device)
            batch_y = torch.stack([self.experience_buffer[i][1] for i in indices]).to(device)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            mean, variance = self.model(batch_X)
            
            # Negative log likelihood loss (accounts for uncertainty)
            loss = self.criterion(mean.squeeze(), batch_y) + 0.5 * variance.mean()
            
            loss.backward()
            self.optimizer.step()
        
        self.update_count += 1
        print(f"Model updated with feedback (update #{self.update_count})")


# ==================== MAIN EXECUTION ====================

def main():
    # Initialize feedback manager
    feedback_manager = FeedbackManager()
    
    # Load the session data
    print("Loading current session data...")
    session_2024 = fastf1.get_session(YEAR, RACE_NUMBER, SESSION_TYPE)
    session_2024.load()
    
    # Get basic lap data
    laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", 
                                    "Sector2Time", "Sector3Time", "Compound",
                                    "TyreLife", "TrackStatus"]].copy()
    laps_2024.dropna(subset=["LapTime"], inplace=True)
    
    # Convert lap and sector times to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()
    
    # Aggregate sector times by driver
    sector_times_2024 = laps_2024.groupby("Driver").agg({
        "Sector1Time (s)": ["mean", "std"],
        "Sector2Time (s)": ["mean", "std"],
        "Sector3Time (s)": ["mean", "std"]
    }).reset_index()
    
    sector_times_2024.columns = ['Driver', 'Sector1Mean', 'Sector1Std',
                                   'Sector2Mean', 'Sector2Std', 
                                   'Sector3Mean', 'Sector3Std']
    
    # Load qualifying data from config
    qualifying_2025 = pd.DataFrame(QUALIFYING_DATA)
    
    # Handle both possible column names for qualifying time
    if "QualifyingTime (s)" not in qualifying_2025.columns and "QualifyingTime" not in qualifying_2025.columns:
        raise KeyError("QUALIFYING_DATA must have either 'QualifyingTime (s)' or 'QualifyingTime' column")
    
    qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(CLEAN_AIR_RACE_PACE)
    
    # Get enhanced features using FastF1 API
    print("Extracting enhanced features from FastF1 API...")
    enhanced_features = prepare_enhanced_features(session_2024, qualifying_2025, None)
    
    # Weather data
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if API_KEY:
        weather_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={WEATHER_LAT}&lon={WEATHER_LON}&appid={API_KEY}&units=metric"
        response = requests.get(weather_url)
        weather_data = response.json()
        
        forecast_dt = datetime.strptime(FORECAST_TIME, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        target_ts = int(forecast_dt.timestamp())
        
        forecast_data = None
        if "hourly" in weather_data:
            forecast_data = min(weather_data["hourly"], key=lambda h: abs(h.get("dt", 0) - target_ts), default=None)
        
        if forecast_data:
            rain_probability = forecast_data.get("pop", 0)
            temperature = forecast_data.get("temp", 25)
            humidity = forecast_data.get("humidity", 50)
            wind_speed = forecast_data.get("wind_speed", 0)
            print(f"\nWeather Forecast: Rain {rain_probability * 100:.1f}%, Temp {temperature}°C, Humidity {humidity}%, Wind {wind_speed} m/s")
        else:
            rain_probability = 0
            temperature = 25
            humidity = 50
            wind_speed = 0
    else:
        rain_probability = 0
        temperature = 25
        humidity = 50
        wind_speed = 0
    
    # Adjust for weather
    wet_performance_factor = 1.15 if rain_probability >= 0.75 else (1.05 if rain_probability >= 0.5 else 1.0)
    
    # Add QualifyingTime column (handle both possible input formats)
    if "QualifyingTime (s)" in qualifying_2025.columns:
        qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * wet_performance_factor
    elif "QualifyingTime" in qualifying_2025.columns:
        # Already has QualifyingTime, just apply weather factor
        qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime"] * wet_performance_factor
    else:
        raise KeyError("No qualifying time column found. Check your race_config_week6.py QUALIFYING_DATA")
    
    # Team performance from config
    max_points = max(TEAM_POINTS.values())
    
    # Handle first race of season (when all teams have 0 points)
    if max_points == 0:
        # Use equal weighting for all teams (or could use last year's standings)
        team_performance_score = {team: 0.5 for team in TEAM_POINTS.keys()}
        print("Note: All teams have 0 points (first race). Using neutral team performance scores.")
    else:
        team_performance_score = {team: points / max_points for team, points in TEAM_POINTS.items()}
    
    qualifying_2025["Team"] = qualifying_2025["Driver"].map(DRIVER_TO_TEAM)
    qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)
    
    # Add weather data
    qualifying_2025["RainProbability"] = rain_probability
    qualifying_2025["Temperature"] = temperature
    qualifying_2025["Humidity"] = humidity
    qualifying_2025["WindSpeed"] = wind_speed
    
    # Merge all data
    merged_data = qualifying_2025.merge(enhanced_features, on="Driver", how="left")
    merged_data = merged_data.merge(sector_times_2024, on="Driver", how="left")
    
    # Filter to valid drivers
    valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
    merged_data = merged_data[valid_drivers]
    
    # Clean up any duplicate columns from merges (pandas adds _x, _y suffixes)
    # We want to keep the QualifyingTime we created, not any from merges
    if "QualifyingTime_x" in merged_data.columns and "QualifyingTime" not in merged_data.columns:
        # Rename QualifyingTime_x to QualifyingTime
        merged_data.rename(columns={"QualifyingTime_x": "QualifyingTime"}, inplace=True)
    if "QualifyingTime_y" in merged_data.columns:
        # Drop the duplicate
        merged_data.drop(columns=["QualifyingTime_y"], inplace=True)
    
    # DEBUG: Print available columns
    print("\n" + "="*80)
    print("DEBUG: Available columns in merged_data (after cleanup):")
    print("="*80)
    print(merged_data.columns.tolist())
    print()
    
    # Prepare features and target
    feature_columns = [
        "QualifyingTime", "RainProbability", "Temperature", "Humidity", "WindSpeed",
        "TeamPerformanceScore", "CleanAirRacePace (s)",
        "Sector1Consistency", "Sector2Consistency", "Sector3Consistency",
        "AvgSpeed", "MaxSpeed", "AvgThrottle", "CleanLapRatio",
        "Sector1Mean", "Sector1Std", "Sector2Mean", "Sector2Std",
        "Sector3Mean", "Sector3Std"
    ]
    
    # DEBUG: Check which columns are missing
    missing_cols = [col for col in feature_columns if col not in merged_data.columns]
    if missing_cols:
        print("="*80)
        print("ERROR: Missing feature columns:")
        print("="*80)
        for col in missing_cols:
            print(f"  - {col}")
        print()
        print("This usually means:")
        print("1. Check your race_config_week6.py file")
        print("2. Make sure QUALIFYING_DATA has 'QualifyingTime (s)' not 'QualifyingTime'")
        print("3. Run diagnostic_tool.py to verify configuration")
        print("="*80)
        raise KeyError(f"Missing columns: {missing_cols}")
    
    X = merged_data[feature_columns]
    y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
    
    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = F1Dataset(X_train, y_train)
    test_dataset = F1Dataset(X_test, y_test)
    
    # Adjust batch size based on dataset size to avoid issues with small batches
    # For small datasets (< 20 samples), use batch size of 2
    batch_size = max(2, min(4, len(X_train) // 4))
    print(f"Using batch size: {batch_size} (train size: {len(X_train)}, test size: {len(X_test)})")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = F1RacePredictorWithUncertainty(
        input_size=X_train.shape[1], 
        hidden_sizes=[256, 128, 64, 32],
        dropout_rate=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25)
    
    # Create online learning wrapper
    online_learner = OnlineLearningWrapper(model, optimizer, criterion)
    
    # Training loop
    num_epochs = 300
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    current_lr = 0.001
    
    print("\nTraining Enhanced Supervised Learning Model with Feedback System...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            mean, variance = model(batch_X)
            
            # Loss combines prediction error and uncertainty
            loss = criterion(mean.squeeze(), batch_y) + 0.1 * variance.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                mean, variance = model(batch_X)
                loss = criterion(mean.squeeze(), batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # Track learning rate changes
        old_lr = current_lr
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"Epoch {epoch+1}: Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'imputer': imputer,
                'feature_columns': feature_columns,
                'val_loss': avg_val_loss,
            }, 'model_checkpoints/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Load best model
    checkpoint = torch.load('model_checkpoints/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make predictions with uncertainty
    model.eval()
    X_scaled_tensor = torch.FloatTensor(X_scaled).to(device)
    
    predictions_mean, predictions_std = model.predict_with_uncertainty(X_scaled_tensor, n_samples=100)
    predictions_mean = predictions_mean.squeeze().cpu().numpy()
    predictions_std = predictions_std.squeeze().cpu().numpy()
    
    merged_data["PredictedRaceTime (s)"] = predictions_mean
    merged_data["PredictionUncertainty (s)"] = predictions_std
    merged_data["ConfidenceInterval_Lower"] = predictions_mean - 1.96 * predictions_std
    merged_data["ConfidenceInterval_Upper"] = predictions_mean + 1.96 * predictions_std
    
    # Sort results
    final_results = merged_data.sort_values(
        by=["PredictedRaceTime (s)", "QualifyingTime"]
    ).reset_index(drop=True)
    
    final_results["PredictedPosition"] = range(1, len(final_results) + 1)
    
    print("\n" + "="*80)
    print("SUPERVISED LEARNING RACE PREDICTION RESULTS WITH UNCERTAINTY")
    print("="*80)
    print(final_results[["Driver", "PredictedPosition", "PredictedRaceTime (s)", 
                         "PredictionUncertainty (s)", "ConfidenceInterval_Lower", 
                         "ConfidenceInterval_Upper"]])
    
    # Display podium with confidence intervals
    podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)", "PredictionUncertainty (s)"]]
    print("\nPredicted Top 3 (with 95% confidence intervals)")
    for i, row in podium.iterrows():
        ci_lower = row["PredictedRaceTime (s)"] - 1.96 * row["PredictionUncertainty (s)"]
        ci_upper = row["PredictedRaceTime (s)"] + 1.96 * row["PredictionUncertainty (s)"]
        position_label = ["1st Place", "2nd Place", "3rd Place"][i]
        print(f"{position_label}: {row['Driver']} - {row['PredictedRaceTime (s)']:.3f}s (±{row['PredictionUncertainty (s)']:.3f}s) [{ci_lower:.3f}s - {ci_upper:.3f}s]")
    
    # Model evaluation
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_mean, _ = model(X_test_tensor)
        y_pred = y_pred_mean.squeeze().cpu().numpy()
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*80}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'='*80}")
        print(f"Mean Absolute Error (MAE):  {mae:.4f} seconds")
        print(f"Root Mean Squared Error:     {rmse:.4f} seconds")
        print(f"R² Score:                    {r2:.4f}")
        print(f"Device used:                 {device}")
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Race Time (s)')
    plt.ylabel('Predicted Race Time (s)')
    plt.title('Prediction vs Actual')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Prediction uncertainty plot
    sorted_idx = np.argsort(predictions_mean)
    plt.errorbar(range(len(predictions_mean)), 
                predictions_mean[sorted_idx], 
                yerr=1.96*predictions_std[sorted_idx],
                fmt='o', capsize=5, capthick=2, alpha=0.7)
    plt.xlabel('Driver (sorted by prediction)')
    plt.ylabel('Predicted Race Time (s)')
    plt.title('Predictions with 95% Confidence Intervals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Feature importance (approximate using gradients)
    model.eval()
    X_sample = torch.FloatTensor(X_scaled[:10]).to(device).requires_grad_(True)
    mean_out, _ = model(X_sample)
    mean_out.sum().backward()
    
    feature_importance = X_sample.grad.abs().mean(dim=0).cpu().numpy()
    feature_names = [col[:15] + '...' if len(col) > 15 else col for col in feature_columns]
    
    top_features_idx = np.argsort(feature_importance)[-10:]
    plt.barh(range(len(top_features_idx)), feature_importance[top_features_idx])
    plt.yticks(range(len(top_features_idx)), [feature_names[i] for i in top_features_idx])
    plt.xlabel('Importance (gradient magnitude)')
    plt.title('Top 10 Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_checkpoints/training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nTraining analysis saved to: model_checkpoints/training_analysis.png")
    
    # Save predictions for feedback
    predictions_for_feedback = [
        {
            "driver": row["Driver"],
            "position": row["PredictedPosition"],
            "time": row["PredictedRaceTime (s)"],
            "uncertainty": row["PredictionUncertainty (s)"]
        }
        for _, row in final_results.iterrows()
    ]
    
    # Save model metadata
    model_metadata = {
        "trained_date": datetime.now().isoformat(),
        "num_features": len(feature_columns),
        "feature_columns": feature_columns,
        "num_training_samples": len(X_train),
        "num_test_samples": len(X_test),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "race_info": {
            "year": YEAR,
            "race_number": RACE_NUMBER,
            "session": SESSION_TYPE
        }
    }
    
    with open('model_checkpoints/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # print("\n" + "="*80)
    # print("FEEDBACK SYSTEM READY")
    # print("="*80)
    # print("After the race, you can update the model with actual results:")
    # print("1. Call feedback_manager.add_race_result() with actual results")
    # print("2. Use online_learner.update_from_feedback() to retrain with new data")
    # print("3. Model will improve predictions for future races")
    
    # Save final results
    final_results.to_csv('model_checkpoints/predictions.csv', index=False)
    print(f"\nPredictions saved to: model_checkpoints/predictions.csv")
    
    # Also save a version with all features for retraining
    predictions_with_features = merged_data.copy()
    predictions_with_features["PredictedPosition"] = final_results["PredictedPosition"]
    predictions_with_features["PredictedRaceTime (s)"] = final_results["PredictedRaceTime (s)"]
    predictions_with_features["PredictionUncertainty (s)"] = final_results["PredictionUncertainty (s)"]
    predictions_with_features.to_csv('model_checkpoints/predictions_with_features.csv', index=False)
    print(f"Full predictions with features saved to: model_checkpoints/predictions_with_features.csv")
    
    return model, online_learner, feedback_manager, final_results, scaler, imputer


if __name__ == "__main__":
    model, online_learner, feedback_manager, results, scaler, imputer = main()