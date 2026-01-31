"""
F1 Race Prediction - Feedback and Model Update System
This script handles post-race feedback and model retraining
Automatically integrates with f1_supervised_feedback_model.py outputs
"""

import fastf1
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
import pickle
import os

from f1_supervised_feedback_model import (
    FeedbackManager, F1RacePredictorWithUncertainty, OnlineLearningWrapper
)


def load_model_components():
    """
    Load all necessary components from the trained model
    
    Returns:
    --------
    tuple: (model, online_learner, scaler, imputer, feature_columns, metadata)
    """
    print("Loading trained model components...")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load('model_checkpoints/best_model.pth', weights_only=False)
        
        # Load metadata
        with open('model_checkpoints/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = metadata['num_features']
        
        model = F1RacePredictorWithUncertainty(
            input_size=input_size,
            hidden_sizes=[256, 128, 64, 32],
            dropout_rate=0.3
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get scalers and feature info
        scaler = checkpoint['scaler']
        imputer = checkpoint['imputer']
        feature_columns = checkpoint['feature_columns']
        
        # Create optimizer and online learner
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        criterion = nn.MSELoss()
        online_learner = OnlineLearningWrapper(model, optimizer, criterion)
        
        print(f" Model loaded successfully")
        print(f"  - Training date: {metadata['trained_date']}")
        print(f"  - Features: {metadata['num_features']}")
        print(f"  - MAE: {metadata['mae']:.4f} seconds")
        print(f"  - R²: {metadata['r2']:.4f}")
        
        return model, online_learner, scaler, imputer, feature_columns, metadata, device
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_predictions_and_config():
    """
    Load predictions and original race configuration
    
    Returns:
    --------
    tuple: (predictions_df, race_year, race_number)
    """
    try:
        # Load predictions with full features for retraining
        predictions_df = pd.read_csv('model_checkpoints/predictions_with_features.csv')
        
        # Load metadata to get race info
        with open('model_checkpoints/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        race_year = metadata['race_info']['year']
        race_number = metadata['race_info']['race_number']
        
        print(f"✓ Loaded predictions for {race_year} Round {race_number}")
        print(f"  - {len(predictions_df)} drivers predicted")
        
        return predictions_df, race_year, race_number
        
    except Exception as e:
        print(f"Error loading predictions: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_actual_race_results(year, race_number, session_type='R'):
    """
    Fetch actual race results from FastF1 API
    
    Parameters:
    -----------
    year : int
        Race year
    race_number : int
        Race number in the season
    session_type : str
        Session type (default 'R' for race)
    
    Returns:
    --------
    list : Actual race results with driver, position, and time
    """
    print(f"Fetching actual results for {year} Round {race_number}...")
    
    try:
        session = fastf1.get_session(year, race_number, session_type)
        session.load()
        
        results = session.results
        
        actual_results = []
        for idx, row in results.iterrows():
            driver_abbr = row['Abbreviation']
            position = row['ClassifiedPosition'] if pd.notna(row['ClassifiedPosition']) else row['Position']
            
            # Handle position - could be numeric or 'R' (Retired), 'D' (Disqualified), etc.
            try:
                position_int = int(float(position)) if pd.notna(position) else None
            except (ValueError, TypeError):
                # Position is not numeric (e.g., 'R' for retired, 'D' for disqualified)
                position_int = None
            
            # Get average lap time for the driver
            driver_laps = session.laps.pick_drivers(driver_abbr)
            if len(driver_laps) > 0:
                avg_lap_time = driver_laps['LapTime'].dt.total_seconds().mean()
            else:
                avg_lap_time = None
            
            actual_results.append({
                "driver": driver_abbr,
                "position": position_int,
                "time": avg_lap_time,
                "status": row['Status'],
                "grid_position": int(row['GridPosition']) if pd.notna(row['GridPosition']) else None,
                "points": row['Points']
            })
        
        return actual_results
    
    except Exception as e:
        print(f"Error fetching race results: {e}")
        return None


def load_predictions(predictions_file='model_checkpoints/predictions.csv'):
    """Load previously saved predictions as list format for compatibility"""
    try:
        predictions_df = pd.read_csv(predictions_file)
        
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append({
                "driver": row['Driver'],
                "position": row['PredictedPosition'],
                "time": row['PredictedRaceTime (s)'],
                "uncertainty": row['PredictionUncertainty (s)']
            })
        
        return predictions
    
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None


def compare_predictions_to_actual(predictions, actual_results):
    """
    Compare predictions to actual results and generate analysis
    
    Parameters:
    -----------
    predictions : list
        List of prediction dictionaries
    actual_results : list
        List of actual result dictionaries
    
    Returns:
    --------
    dict : Comparison analysis with errors and insights
    """
    pred_dict = {p['driver']: p for p in predictions}
    actual_dict = {a['driver']: a for a in actual_results if a['position'] is not None}
    
    comparison = []
    position_errors = []
    time_errors = []
    
    for driver in pred_dict.keys():
        if driver in actual_dict:
            pred_pos = pred_dict[driver]['position']
            actual_pos = actual_dict[driver]['position']
            pred_time = pred_dict[driver]['time']
            actual_time = actual_dict[driver]['time']
            
            pos_error = abs(pred_pos - actual_pos)
            time_error = abs(pred_time - actual_time) if actual_time else None
            
            comparison.append({
                'driver': driver,
                'predicted_position': pred_pos,
                'actual_position': actual_pos,
                'position_error': pos_error,
                'predicted_time': pred_time,
                'actual_time': actual_time,
                'time_error': time_error,
                'prediction_uncertainty': pred_dict[driver]['uncertainty']
            })
            
            position_errors.append(pos_error)
            if time_error:
                time_errors.append(time_error)
    
    comparison_df = pd.DataFrame(comparison)
    
    # Calculate metrics
    mean_position_error = np.mean(position_errors)
    mean_time_error = np.mean(time_errors) if time_errors else None
    
    # Check how many drivers were in top 3 correctly
    top3_correct = 0
    for driver in actual_dict.keys():
        if driver in pred_dict:
            if actual_dict[driver]['position'] <= 3 and pred_dict[driver]['position'] <= 3:
                top3_correct += 1
    
    # Check podium exact match
    actual_podium = sorted([d for d in actual_dict.values() if d['position'] <= 3], 
                          key=lambda x: x['position'])
    pred_podium = sorted([p for p in predictions if p['position'] <= 3], 
                        key=lambda x: x['position'])
    
    podium_exact_match = all(
        actual_podium[i]['driver'] == pred_podium[i]['driver']
        for i in range(min(3, len(actual_podium), len(pred_podium)))
    )
    
    analysis = {
        'mean_position_error': mean_position_error,
        'mean_time_error': mean_time_error,
        'top3_drivers_correct': top3_correct,
        'podium_exact_match': podium_exact_match,
        'comparison_table': comparison_df,
        'total_drivers_compared': len(comparison)
    }
    
    return analysis


def retrain_with_actual_results(model, online_learner, scaler, imputer, 
                                 feature_columns, actual_results, predictions_df, device):
    """
    Retrain model using actual race results as new training data
    
    Parameters:
    -----------
    model : F1RacePredictorWithUncertainty
        The trained model
    online_learner : OnlineLearningWrapper
        Online learning wrapper
    scaler : StandardScaler
        Feature scaler
    imputer : SimpleImputer
        Feature imputer
    feature_columns : list
        List of feature column names
    actual_results : list
        Actual race results
    predictions_df : DataFrame
        Original predictions with features
    device : torch.device
        Device to use for training
        
    Returns:
    --------
    bool : Success status
    """
    
    try:
        # Extract drivers who finished with valid times
        valid_results = [r for r in actual_results if r['time'] is not None and r['position'] is not None]
        
        if len(valid_results) < 5:
            print(f" Only {len(valid_results)} drivers with valid results - need at least 5 for retraining")
            return False
        
        print(f"Using {len(valid_results)} drivers with valid race results for retraining...")
        
        # Match actual results with prediction features
        X_new = []
        y_new = []
        
        for result in valid_results:
            driver = result['driver']
            
            # Find this driver in predictions_df to get their features
            driver_row = predictions_df[predictions_df['Driver'] == driver]
            
            if len(driver_row) > 0:
                # Extract the same features used in training
                # We need to reconstruct the feature vector
                driver_features = {}
                for col in feature_columns:
                    if col in driver_row.columns:
                        driver_features[col] = driver_row[col].values[0]
                    else:
                        driver_features[col] = 0  # Default value for missing features
                
                feature_vector = [driver_features[col] for col in feature_columns]
                X_new.append(feature_vector)
                y_new.append(result['time'])
        
        if len(X_new) == 0:
            print(" Could not match actual results with prediction features")
            return False
        
        # Convert to numpy arrays
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        print(f"Prepared {len(X_new)} training samples from actual race results")
        
        # Preprocess features (same as training)
        X_new_imputed = imputer.transform(X_new)
        X_new_scaled = scaler.transform(X_new_imputed)
        
        # Convert to tensors and add to experience buffer
        print("Adding experiences to online learner...")
        for i in range(len(X_new_scaled)):
            x_tensor = torch.FloatTensor(X_new_scaled[i]).to(device)
            y_tensor = torch.FloatTensor([y_new[i]]).to(device)
            online_learner.add_experience(x_tensor, y_tensor)
        
        # Update model with new data (no need to pass X and y again, they're in the buffer)
        print("Updating model weights...")
        
        # Train on the experience buffer
        model.train()
        for epoch in range(20):
            if len(online_learner.experience_buffer) < 2:
                break
                
            # Sample from experience buffer
            batch_size = min(2, len(online_learner.experience_buffer))
            indices = np.random.choice(len(online_learner.experience_buffer), 
                                      size=batch_size, 
                                      replace=False)
            
            batch_X = torch.stack([online_learner.experience_buffer[i][0] for i in indices])
            batch_y = torch.stack([online_learner.experience_buffer[i][1] for i in indices])
            
            online_learner.optimizer.zero_grad()
            mean, variance = model(batch_X)
            loss = online_learner.criterion(mean.squeeze(), batch_y.squeeze()) + 0.1 * variance.mean()
            loss.backward()
            online_learner.optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/20, Loss: {loss.item():.4f}")
        
        online_learner.update_count += 1
        print(f"Model updated (update #{online_learner.update_count})")
        
        # Save updated model
        print("Saving updated model...")
        torch.save({
            'epoch': 'post_race_update',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': online_learner.optimizer.state_dict(),
            'scaler': scaler,
            'imputer': imputer,
            'feature_columns': feature_columns,
            'update_info': {
                'update_date': datetime.now().isoformat(),
                'num_new_samples': len(X_new),
                'drivers_updated': [r['driver'] for r in valid_results]
            }
        }, 'model_checkpoints/updated_model.pth')
        
        # Also update the best model
        torch.save({
            'epoch': 'post_race_update',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': online_learner.optimizer.state_dict(),
            'scaler': scaler,
            'imputer': imputer,
            'feature_columns': feature_columns,
        }, 'model_checkpoints/best_model.pth')
        
        print(f" Model updated and saved successfully!")
        print(f"  - Added {len(X_new)} new training examples")
        print(f"  - Updated drivers: {', '.join([r['driver'] for r in valid_results[:5]])}...")
        
        return True
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_feedback_report(analysis, save_path='feedback_data/race_analysis.txt'):
    """Generate a detailed feedback report"""
    
    report = []
    report.append("="*80)
    report.append("RACE PREDICTION FEEDBACK REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("OVERALL PERFORMANCE")
    report.append("-"*80)
    report.append(f"Mean Position Error:      {analysis['mean_position_error']:.2f} positions")
    if analysis['mean_time_error']:
        report.append(f"Mean Time Error:          {analysis['mean_time_error']:.3f} seconds")
    report.append(f"Top 3 Drivers Correct:    {analysis['top3_drivers_correct']}/3")
    report.append(f"Podium Exact Match:       {'Yes' if analysis['podium_exact_match'] else 'No'}")
    report.append(f"Total Drivers Compared:   {analysis['total_drivers_compared']}")
    report.append("")
    
    report.append("DETAILED COMPARISON")
    report.append("-"*80)
    
    comparison_df = analysis['comparison_table']
    comparison_df = comparison_df.sort_values('actual_position')
    
    for _, row in comparison_df.iterrows():
        status = "OK" if row['position_error'] == 0 else f"±{int(row['position_error'])}"
        report.append(f"P{int(row['actual_position']):2d} {row['driver']:4s} "
                     f"(Predicted: P{int(row['predicted_position']):2d}) "
                     f"[{status}] "
                     f"Time Error: {row['time_error']:.3f}s" if pd.notna(row['time_error']) else "")
    
    report.append("")
    report.append("="*80)
    
    report_text = "\n".join(report)
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {save_path}")


def main_feedback_workflow(year=None, race_number=None, retrain_model=False):
    """
    Main workflow for post-race feedback and model update
    Automatically loads race info from trained model if not provided
    
    Parameters:
    -----------
    year : int, optional
        Race year (if None, loads from model metadata)
    race_number : int, optional
        Race number (if None, loads from model metadata)
    retrain_model : bool
        Whether to retrain the model with actual results
    """
    
    print("="*80)
    print("F1 RACE PREDICTION - FEEDBACK AND UPDATE SYSTEM")
    print("="*80)
    print()
    
    # Load predictions and race info
    predictions_df, auto_year, auto_race_number = load_predictions_and_config()
    
    if predictions_df is None:
        print("Error: Could not load predictions. Please run f1_supervised_feedback_model.py first.")
        return
    
    # Use auto-detected race info if not provided
    race_year = year if year is not None else auto_year
    race_number_to_use = race_number if race_number is not None else auto_race_number
    
    print(f"Analyzing race: {race_year} Round {race_number_to_use}")
    print()
    
    # Convert predictions to list format
    predictions = load_predictions()
    
    # Initialize feedback manager
    feedback_manager = FeedbackManager()
    
    # Get actual results
    actual_results = get_actual_race_results(race_year, race_number_to_use)
    
    if actual_results is None:
        print("Error: Could not fetch actual results.")
        return
    
    print(f"\nFetched results for {len(actual_results)} drivers")
    
    # Compare predictions to actual
    print("\nAnalyzing prediction accuracy...")
    analysis = compare_predictions_to_actual(predictions, actual_results)
    
    # Generate report
    generate_feedback_report(analysis)
    
    # Store feedback
    race_id = f"{race_year}_R{race_number_to_use}"
    metadata = {
        "year": race_year,
        "race_number": race_number_to_use,
        "analysis_date": datetime.now().isoformat()
    }
    
    feedback_entry = feedback_manager.add_race_result(
        race_id, predictions, actual_results, metadata
    )
    
    print(f"\nFeedback stored for race: {race_id}")
    
    # Optionally retrain model with actual results
    if retrain_model:
        print("\n" + "="*80)
        print("RETRAINING MODEL WITH ACTUAL RESULTS")
        print("="*80)
        
        components = load_model_components()
        if components:
            model, online_learner, scaler, imputer, feature_columns, model_metadata, device = components
            
            # Prepare actual results for retraining
            print("\nPreparing actual race data for model update...")
            retrain_success = retrain_with_actual_results(
                model, online_learner, scaler, imputer, 
                feature_columns, actual_results, predictions_df, device
            )
            
            if retrain_success:
                print("  Model successfully updated with actual race results!")
                print("  The model will now make better predictions for future races.")
            else:
                print("  Model update skipped (insufficient data or error)")
    
    # Get performance trends
    trends = feedback_manager.get_performance_trends()
    if trends:
        print("\n" + "="*80)
        print("PERFORMANCE TRENDS")
        print("="*80)
        driver_count = 0
        for driver, trend_data in sorted(trends.items(), key=lambda x: np.mean(x[1]['position_errors']) if x[1]['position_errors'] else 999):
            if trend_data['position_errors']:
                avg_pos_error = np.mean(trend_data['position_errors'])
                races_analyzed = len(trend_data['position_errors'])
                print(f"  {driver}: Avg Position Error = {avg_pos_error:.2f} ({races_analyzed} race(s))")
                driver_count += 1
                if driver_count >= 10:  # Show top 10
                    break
    
    # print("\n" + "="*80)
    # print("FEEDBACK WORKFLOW COMPLETE")
    # print("="*80)
    # print("\nNext steps:")
    # print("1. Review the detailed analysis report in feedback_data/race_analysis.txt")
    # print("2. Identify patterns in prediction errors")
    # print("3. Run with retrain_model=True to improve the model:")
    # print("   python -c \"from f1_feedback_system import main_feedback_workflow; main_feedback_workflow(retrain_model=True)\"")
    # print("4. Model will automatically get better with more race feedback!")
    
    return analysis


if __name__ == "__main__":
    # The script now auto-detects race info from the trained model
    # You can optionally override:
    # main_feedback_workflow(year=2024, race_number=6)
    
    # Or just run with auto-detection:
    print("Running feedback analysis with auto-detected race information...")
    print("(Race info will be loaded from model_checkpoints/model_metadata.json)")
    print()
    
    # Run without retraining (just analysis)
    # main_feedback_workflow()
    
    # To run with retraining, uncomment:
    main_feedback_workflow(retrain_model=True)