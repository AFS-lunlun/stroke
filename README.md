# KAN and DNN Model Prediction Tool

This project provides tools for loading and using KAN (Kolmogorov-Arnold Network from the PyKAN project) and DNN (Deep Neural Network) models for prediction tasks.

## Features

- Support for loading KAN and DNN models for predictions
- Automatic data normalization
- Support for both classification and regression problems
- Model evaluation and visualization capabilities
- Support for CSV and Excel data formats

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### DNN Model Prediction

```bash
python run_dnn_model.py \
    --model_folder <model_directory> \
    --train_file <training_data_file> \
    --data_file <prediction_data_file> \
    [--output_file <predictions.csv>]
```

Parameters:
- `--model_folder/-m`: Directory containing config_param.json and .h5 model files
- `--train_file/-t`: Original training data (CSV or Excel) for normalization parameters
- `--data_file/-d`: New data for prediction (CSV or Excel)
- `--output_file/-o`: Optional, path to save prediction results as CSV; if omitted, prints to stdout

### KAN Model Prediction

```bash
python run_kan_model.py \
    --model_folder <model_directory> \
    --train_file <training_data_file> \
    --data_file <prediction_data_file> \
    [--output_file <predictions.csv>]
```

Parameters are the same as for DNN model.

## File Descriptions

- `run_dnn_model.py`: DNN model prediction script
- `run_kan_model.py`: KAN model prediction script
- `requirements.txt`: Project dependencies

## Data Format Requirements

- Training data: Last column is the label, all other columns are features
- Prediction data: All columns are features, matching the number of feature columns in training data

## Notes

- Ensure data files have no missing values
- Data normalization method is automatically read from the model configuration
