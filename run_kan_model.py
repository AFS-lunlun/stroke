#!/usr/bin/env python3
"""
run_kan_model: KAN 模型预测新数据

用法：
    python run_kan_model.py \
        --model_folder <模型目录> \
        --train_file <训练数据文件> \
        --data_file <待预测数据文件> \
        [--output_file <预测结果.csv>]

描述：
    加载已保存的回归或分类 KAN/ClsKan 模型及其配置，
    从原始训练数据计算归一化参数，
    对新数据应用相同的归一化，并生成预测结果。
    回归模型的输出会按原始标签尺度反归一化。

参数：
    --model_folder/-m   包含 config_param.json 和 model.0_* 文件的目录
    --train_file/-t     原始训练数据（CSV 或 Excel），用于计算归一化参数
    --data_file/-d      待预测的新数据（CSV 或 Excel）
    --output_file/-o    可选，保存预测结果的 CSV 文件路径；若省略则打印到标准输出

示例：
    python run_kan_model.py \
        --model_folder model_1 \
        --train_file data/train.xlsx \
        --data_file data/new_samples.csv \
        --output_file data/preds.csv
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from kan import KAN

class ClsKan(KAN):
    def forward(self, x, singularity_avoiding=False, y_th=10):
        return torch.nn.Sigmoid()(super().forward(x, singularity_avoiding, y_th))

    @classmethod
    def loadckpt(cls, path='model'):
        """
        Load a KAN checkpoint and cast it to cls, ensuring sigmoid forward.
        """
        model = KAN.loadckpt(path)
        model.__class__ = cls
        return model
    
def input_normalization(data ,method='none'):
    # normalize each column. 
    norm_op_dict = {
        # (data-mean)/std
        'ext':lambda data: data / np.max(np.abs(data), axis=0),
        'avg': lambda data: data - np.mean(data, axis=0),
        'avgext': lambda data: (data - np.mean(data, axis=0))/np.max(np.abs(data), axis=0),
        'avgstd': lambda data: (data - np.mean(data,  axis=0))/np.std(data, axis=0),
        'none': lambda data: data,
    }
    if data is None:
        return None

    result = norm_op_dict[method](data)

    result[np.isnan(result)] = 0

    return {
        'result':result,
        'param':{
            'method':method,
            'max':np.max(np.abs(data)),
            'mean':np.mean(data, axis=0),
            'std':np.std(data, axis=0)
        }
    }
    

def input_normalization_param(data, param):
    """Normalize data using provided normalization parameters."""
    norm_op_dict = {
        'ext': lambda data: data / param.get('max', 1),
        'avg': lambda data: data - param.get('mean', 0),
        'avgext': lambda data: (data - param.get('mean', 0)) / param.get('max', 1),
        'avgstd': lambda data: (data - param.get('mean', 0)) / param.get('std', 1),
        'none': lambda data: data,
    }
    if data is None:
        return None
    result = norm_op_dict[param['method']](data)
    result[np.isnan(result)] = 0
    return result


def load_model(model_folder):
    # Load config
    cfg_path = os.path.join(model_folder, 'config_param.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, 'r') as f:
        config = json.load(f)
    # Load model checkpoint (path without suffix)
    ckpt_base = os.path.join(model_folder, 'model.0')
    if config.get('problem_type') == 'classification':
        model = ClsKan.loadckpt(path=ckpt_base)
        print('这是一个分类模型')
    else:
        model = KAN.loadckpt(path=ckpt_base)
        print('这是一个回归模型')
    model.eval()
    print(model)
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Predict new data using a trained KAN model')
    parser.add_argument('--model_folder', '-m', required=True,
                        help='Path to model directory (contains config_param.json and model.0_*)')
    parser.add_argument('--data_file', '-d', required=True,
                        help='Path to new data file (CSV or Excel)')
    parser.add_argument('--train_file', '-t', required=True,
                        help='Path to training data file (CSV or Excel) for normalization parameters')
    parser.add_argument('--output_file', '-o', default=None,
                        help='Path to save predictions (CSV). If omitted, prints to stdout.')
    args = parser.parse_args()

    model, config = load_model(args.model_folder)

    # load training data for normalization
    if args.train_file.lower().endswith('.csv'):
        train_df = pd.read_csv(args.train_file)
    elif args.train_file.lower().endswith(('.xls', '.xlsx')):
        train_df = pd.read_excel(args.train_file)
    else:
        raise ValueError('Unsupported train file format. Use CSV or Excel.')
    train_feature_cols = train_df.columns[:-2].tolist()
    train_features = train_df[train_feature_cols].to_numpy().astype(np.float32)
    norm_method = config.get('norm_method', 'none')
    norm_params = input_normalization(train_features, norm_method)['param']

    # compute y_scale for regression
    if config.get('problem_type') != 'classification':
        label_col = train_df.columns[-1]
        train_labels = train_df[label_col].to_numpy().astype(np.float32)
        y_scale = np.max(np.abs(train_labels))
    else:
        y_scale = 1.0

    # Load new data
    if args.data_file.lower().endswith('.csv'):
        df = pd.read_csv(args.data_file)
    elif args.data_file.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(args.data_file)
    else:
        raise ValueError('Unsupported file format. Use CSV or Excel.')

    # Assume all columns are features (no label column)
    feature_cols = df.columns.tolist()
    X = df[feature_cols].to_numpy().astype(np.float32)

    # Normalize new data using training normalization parameters
    norm_method = config.get('norm_method', 'none')
    X_norm = input_normalization_param(X, norm_params)

    # Predict
    X_tensor = torch.tensor(X_norm, dtype=torch.float)
    with torch.no_grad():
        preds = model(X_tensor)
        preds = preds.numpy().squeeze()
        # rescale predictions for regression
        if config.get('problem_type') != 'classification':
            preds = preds * y_scale

    # Append predictions
    df['prediction'] = preds

    # Save or display
    if args.output_file:
        df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    else:
        print(df)

if __name__ == '__main__':
    main()
