#!/usr/bin/env python3
"""
run_dnn_model: DNN 模型预测新数据

用法：
    python run_dnn_model.py \
        --model_folder <模型目录> \
        --train_file <训练数据文件> \
        --data_file <待预测数据文件> \
        [--output_file <预测结果.csv>]

描述：
    加载已保存的回归或分类 DNN 模型及其配置，
    从原始训练数据计算归一化参数，
    对新数据应用相同的归一化，并生成预测结果。
    回归模型的输出会按原始标签尺度反归一化。

参数：
    --model_folder/-m   包含 config_param.json 和 model.h5 文件的目录
    --train_file/-t     原始训练数据（CSV 或 Excel），用于计算归一化参数
    --data_file/-d      待预测的新数据（CSV 或 Excel）
    --output_file/-o    可选，保存预测结果的 CSV 文件路径；若省略则打印到标准输出

示例：
    python run_dnn_model.py \
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
import tensorflow as tf
import traceback

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
    """加载模型和配置文件
    
    Args:
        model_folder: 模型目录路径
        
    Returns:
        model_wrapper: 模型包装类实例
        config: 配置信息字典
    """
    # 加载配置文件
    cfg_path = os.path.join(model_folder, 'config_param.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件未找到: {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        config = json.load(f)
    
    # 查找模型文件
    h5_files = [f for f in os.listdir(model_folder) if f.endswith('.h5')]
    if not h5_files:
        raise FileNotFoundError(f"在 {model_folder} 中未找到.h5模型文件")
    
    # 使用找到的第一个.h5文件
    model_path = os.path.join(model_folder, h5_files[0])
    print(f"使用模型文件: {model_path}")
    
    try:
        # 禁用eager execution以兼容TF1模型
        tf.compat.v1.disable_eager_execution()
        
        # 加载模型
        model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf})
        
        # 创建一个包装类来处理预测
        class DNNModel:
            def __init__(self, model, is_classification=False):
                self.model = model
                self.is_classification = is_classification
            
            def predict(self, x):
                # 确保输入数据格式正确
                if isinstance(x, pd.DataFrame):
                    x = x.values
                if isinstance(x, np.ndarray):
                    if len(x.shape) == 1:
                        x = x.reshape(1, -1)
                return self.model.predict(x)
        
        # 判断模型类型
        is_classification = config.get('problem_type') == 'classification'
        
        if is_classification:
            print('这是一个分类模型')
        else:
            print('这是一个回归模型')
        
        # 输出模型信息
        if hasattr(model, 'summary'):
            model.summary()
        
        return DNNModel(model, is_classification), config
        
    except Exception as e:
        print(f"加载模型时出错:")
        traceback.print_exc()
        # raise RuntimeError(f"无法加载模型: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Predict new data using a trained DNN model')
    parser.add_argument('--model_folder', '-m', required=True,
                        help='Path to model directory (contains config_param.json and model.h5)')
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
    
    # 假设最后一列是标签列
    # 去掉空列
    train_df = train_df.dropna(axis=1, how='all')
    train_feature_cols = train_df.columns[:-1].tolist()
    train_features = train_df[train_feature_cols].to_numpy().astype(np.float32)
    
    # 获取标准化方法，默认为'none'
    norm_method = config.get('net_property', {}).get('input_norm', 'none')
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

    # 假设所有列都是特征（没有标签列）
    X = df.iloc[:, :len(train_feature_cols)].to_numpy().astype(np.float32)

    # Normalize new data using training normalization parameters
    X_norm = input_normalization_param(X, norm_params)

    # Predict
    preds = model.predict(X_norm).squeeze()
    
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

