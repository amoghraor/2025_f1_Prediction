# F1 Race Prediction - Supervised Learning Model with Feedback System

## Overview

This enhanced F1 race prediction system uses **supervised learning** with a **feedback loop** to continuously improve predictions based on actual race results. The system integrates deeply with the FastF1 API to extract accurate, real-time race data.

## Key Features

### 1. **Supervised Learning Model**
- Deep Neural Network with uncertainty estimation
- Predicts race times with confidence intervals
- Multi-layer architecture (256→128→64→32 neurons)
- Batch normalization and dropout for regularization

### 2. **Feedback System**
- **Post-race validation**: Compare predictions with actual results
- **Error tracking**: Track prediction errors over time
- **Performance trends**: Analyze model improvement across races
- **Manual corrections**: Add expert knowledge adjustments

### 3. **Online Learning**
- **Experience replay buffer**: Stores past predictions and outcomes
- **Incremental updates**: Retrain model with new race data
- **Continuous improvement**: Model gets better after each race

### 4. **Enhanced Features from FastF1 API**
- **Sector consistency**: Standard deviation of sector times
- **Speed metrics**: Average and maximum speeds
- **Throttle analysis**: Average throttle application
- **Tire compound data**: Dominant tire compounds used
- **Track status**: Clean lap ratio (avoiding yellow flags)
- **Historical data**: Can load multiple years of data

### 5. **Uncertainty Quantification**
- **Monte Carlo dropout**: Estimates prediction uncertainty
- **Confidence intervals**: 95% confidence bounds for each prediction
- **Risk assessment**: Identifies high-uncertainty predictions



## Installation

```bash
# Install required packages
pip install fastf1 pandas numpy scikit-learn torch matplotlib requests python-dotenv --break-system-packages

# Enable F1 cache
mkdir -p f1_cache model_checkpoints feedback_data
```

## Usage

### Step 1: Initial Prediction

```python
# Run the main prediction model
python f1_supervised_feedback_model.py
```

**Output:**
- Predicted race positions with uncertainty
- Confidence intervals for each driver
- Training performance metrics
- Visualization plots

### Step 2: Post-Race Feedback (After the race)

```python
# Run the feedback system after the race
python f1_feedback_system.py
```

**Output:**
- Comparison of predictions vs actual results
- Position and time error analysis
- Detailed feedback report
- Model performance trends

### Step 3: Model Updates

The model automatically improves through:
1. **Stored feedback**: All race results are saved
2. **Experience replay**: Past races help train the model
3. **Incremental learning**: New data updates weights without full retraining

## Configuration

Create a `race_config_week.py` file which has similar template to `race_config_tempelate.py` and use the `diagnostic_tool.py` to confirm if the config file is accepted by the code or not.



## Environment Variables

Create a `.env` file:

```bash
OPENWEATHER_API_KEY=your_api_key_here
```

## Feature Engineering Details

### Core Features (20 total)

1. **Qualifying Performance**
   - Raw qualifying time
   - Weather-adjusted qualifying time

2. **Weather Conditions**
   - Rain probability
   - Temperature
   - Humidity
   - Wind speed

3. **Team Performance**
   - Team performance score (normalized points)

4. **Race Pace**
   - Clean air race pace from practice

5. **Sector Analysis** (from FastF1)
   - Sector 1/2/3 mean times
   - Sector 1/2/3 consistency (std deviation)

6. **Speed Metrics** (from FastF1)
   - Average speed
   - Maximum speed
   - Average throttle application

7. **Track Behavior**
   - Clean lap ratio (avoiding yellow flags)



### Loss Function

```python
# Combined loss: prediction error + uncertainty regularization
loss = MSE(predicted_mean, actual_time) + 0.1 * mean(predicted_variance)
```

### Training Process

- **Epochs**: 300 (with early stopping)
- **Batch Size**: 4
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Early Stopping**: Patience of 50 epochs
- **Gradient Clipping**: Max norm = 1.0


## Performance Metrics

The system tracks multiple metrics:

### Training Metrics
- **MAE** (Mean Absolute Error): Average time error in seconds
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R² Score**: Proportion of variance explained

### Prediction Metrics (Post-race)
- **Mean Position Error**: Average positions off
- **Mean Time Error**: Average time difference in seconds
- **Top 3 Accuracy**: How many podium drivers predicted correctly
- **Podium Exact Match**: Whether entire podium was correct

## Contributing

To improve the model:
1. Add new features from FastF1 API
2. Experiment with different architectures
3. Tune hyperparameters
4. Provide feedback on predictions
5. Share insights from error analysis

## License

This project uses the FastF1 API which is subject to its own license terms.

## Acknowledgments

- **FastF1**: Excellent F1 data API
- **PyTorch**: Deep learning framework
- **Scikit-learn**: Machine learning utilities
- **OpenWeatherMap**: Weather data API

## Contact & Support

For questions or issues:
1. Check FastF1 documentation: https://docs.fastf1.dev/
2. Review error logs in console
3. Verify race configuration settings
4. Ensure API keys are valid

## Credits

Built independently but inspired by [@mar-antaya](https://github.com/mar-antaya/2025_f1_predictions.git). Data sources: FastF1 and OpenWeatherMap.

---

**Happy Racing! 🏎️**




