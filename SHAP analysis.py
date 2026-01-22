#!/usr/bin/env python3
"""
run_shap_explanation: 使用 SHAP 解释 DNN 模型预测

用法：
    python run_shap_explanation.py \
        --model_folder <模型目录> \
        --train_file <训练数据文件> \
        --data_file <待解释数据文件> \
        [--output_dir <输出目录>] \
        [--nsamples <背景样本数>] \
        [--plot_type <可视化类型>]

描述：
    加载已保存的回归或分类 DNN 模型及其配置，
    从原始训练数据计算归一化参数，
    对新数据应用相同的归一化，并使用 SHAP 值解释模型预测。
    生成 SHAP 可视化图表和特征重要性数据。

参数：
    --model_folder/-m   包含 config_param.json 和 model.h5 文件的目录
    --train_file/-t     原始训练数据（CSV 或 Excel），用于计算归一化参数和背景数据
    --data_file/-d      待解释的新数据（CSV 或 Excel）
    --output_dir/-o     可选，保存 SHAP 结果的目录；若省略则使用当前目录下的 shap_output
    --nsamples/-n       可选，用于 SHAP 解释的背景样本数量（默认：100）
    --plot_type/-p      可选，可视化类型：summary/waterfall/bar/force（默认：summary）

示例：
    python run_shap_explanation.py \
        --model_folder model_1 \
        --train_file data/train.xlsx \
        --data_file data/new_samples.csv \
        --output_dir shap_results \
        --nsamples 200 \
        --plot_type summary
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import traceback
import matplotlib.pyplot as plt
import shap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def input_normalization(data, method='none'):
    # normalize each column. 
    norm_op_dict = {
        # (data-mean)/std
        'ext': lambda data: data / np.max(np.abs(data), axis=0),
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
        'result': result,
        'param': {
            'method': method,
            'max': np.max(np.abs(data)),
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
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
        raise RuntimeError(f"无法加载模型: {str(e)}")


def create_shap_explainer(model, background_data, nsamples=100):
    """创建 SHAP 解释器
    
    Args:
        model: 训练好的模型
        background_data: 背景数据集（用于计算 SHAP 值）
        nsamples: 背景样本数量
        
    Returns:
        explainer: SHAP 解释器
    """
    print(f"创建 SHAP 解释器，使用 {nsamples} 个背景样本...")
    
    # 如果背景数据太大，进行采样
    if len(background_data) > nsamples:
        indices = np.random.choice(len(background_data), nsamples, replace=False)
        background_sample = background_data[indices]
    else:
        background_sample = background_data
    
    # 使用 DeepExplainer 用于深度神经网络
    try:
        explainer = shap.DeepExplainer(model.model, background_sample)
        print("使用 DeepExplainer")
    except Exception as e:
        print(f"DeepExplainer 失败，尝试使用 KernelExplainer: {str(e)}")
        # 如果 DeepExplainer 失败，使用 KernelExplainer
        def model_predict(x):
            return model.predict(x)
        explainer = shap.KernelExplainer(model_predict, background_sample)
        print("使用 KernelExplainer")
    
    return explainer


def generate_shap_plots(explainer, data_to_explain, feature_names, output_dir, plot_type='summary'):
    """生成 SHAP 可视化图表
    
    Args:
        explainer: SHAP 解释器
        data_to_explain: 待解释的数据
        feature_names: 特征名称列表
        output_dir: 输出目录
        plot_type: 可视化类型
    """
    print(f"计算 SHAP 值...")
    shap_values = explainer.shap_values(data_to_explain)
    
    # 如果是分类模型，shap_values 可能是列表
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 取第一个类别的 SHAP 值
    
    print(f"SHAP 值形状: {shap_values.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 SHAP 值到 CSV
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)
    print(f"SHAP 值已保存到: {os.path.join(output_dir, 'shap_values.csv')}")
    
    # 计算平均绝对 SHAP 值（特征重要性）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print(f"特征重要性已保存到: {os.path.join(output_dir, 'feature_importance.csv')}")
    
    # 生成可视化图表
    print(f"生成 {plot_type} 类型的可视化图表...")
    
    if plot_type == 'summary':
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, data_to_explain, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary plot 已保存到: {os.path.join(output_dir, 'shap_summary_plot.png')}")
        
        # Summary bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, data_to_explain, feature_names=feature_names, 
                         plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary bar plot 已保存到: {os.path.join(output_dir, 'shap_summary_bar.png')}")
    
    elif plot_type == 'waterfall':
        # Waterfall plot (for single sample)
        if len(data_to_explain) == 1:
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                  base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                                  data=data_to_explain[0],
                                                  feature_names=feature_names), show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_waterfall_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Waterfall plot 已保存到: {os.path.join(output_dir, 'shap_waterfall_plot.png')}")
        else:
            print("Waterfall plot 仅适用于单个样本，生成第一个样本的图表...")
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                  base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                                  data=data_to_explain[0],
                                                  feature_names=feature_names), show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_waterfall_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Waterfall plot 已保存到: {os.path.join(output_dir, 'shap_waterfall_plot.png')}")
    
    elif plot_type == 'bar':
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap.Explanation(values=shap_values, 
                                        data=data_to_explain,
                                        feature_names=feature_names), show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bar plot 已保存到: {os.path.join(output_dir, 'shap_bar_plot.png')}")
    
    elif plot_type == 'force':
        # Force plot (for single sample)
        if len(data_to_explain) == 1:
            shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                           shap_values[0], 
                           data_to_explain[0],
                           feature_names=feature_names,
                           matplotlib=True,
                           show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_force_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Force plot 已保存到: {os.path.join(output_dir, 'shap_force_plot.png')}")
        else:
            print("Force plot 仅适用于单个样本，生成第一个样本的图表...")
            shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                           shap_values[0], 
                           data_to_explain[0],
                           feature_names=feature_names,
                           matplotlib=True,
                           show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shap_force_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Force plot 已保存到: {os.path.join(output_dir, 'shap_force_plot.png')}")
    
    # 始终生成 summary plot 作为默认
    if plot_type != 'summary':
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, data_to_explain, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary plot 已保存到: {os.path.join(output_dir, 'shap_summary_plot.png')}")


def main():
    parser = argparse.ArgumentParser(description='Explain DNN model predictions using SHAP')
    parser.add_argument('--model_folder', '-m', required=True,
                        help='Path to model directory (contains config_param.json and model.h5)')
    parser.add_argument('--data_file', '-d', required=True,
                        help='Path to data file to explain (CSV or Excel)')
    parser.add_argument('--train_file', '-t', required=True,
                        help='Path to training data file (CSV or Excel) for normalization parameters and background data')
    parser.add_argument('--output_dir', '-o', default='shap_output',
                        help='Directory to save SHAP results (default: shap_output)')
    parser.add_argument('--nsamples', '-n', type=int, default=100,
                        help='Number of background samples for SHAP (default: 100)')
    parser.add_argument('--plot_type', '-p', default='summary',
                        choices=['summary', 'waterfall', 'bar', 'force'],
                        help='Type of visualization (default: summary)')
    args = parser.parse_args()

    # 加载模型
    model, config = load_model(args.model_folder)

    # 加载训练数据用于归一化和背景数据
    if args.train_file.lower().endswith('.csv'):
        train_df = pd.read_csv(args.train_file)
    elif args.train_file.lower().endswith(('.xls', '.xlsx')):
        train_df = pd.read_excel(args.train_file)
    else:
        raise ValueError('Unsupported train file format. Use CSV or Excel.')
    
    # 去掉空列
    train_df = train_df.dropna(axis=1, how='all')
    train_feature_cols = train_df.columns[:-1].tolist()
    train_features = train_df[train_feature_cols].to_numpy().astype(np.float32)
    
    # 获取标准化方法，默认为'none'
    norm_method = config.get('net_property', {}).get('input_norm', 'none')
    norm_params = input_normalization(train_features, norm_method)['param']
    
    # 归一化训练数据作为背景数据
    train_features_norm = input_normalization_param(train_features, norm_params)

    # 加载待解释的数据
    if args.data_file.lower().endswith('.csv'):
        df = pd.read_csv(args.data_file)
    elif args.data_file.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(args.data_file)
    else:
        raise ValueError('Unsupported file format. Use CSV or Excel.')

    # 假设所有列都是特征（或与训练数据列数相同）
    if len(df.columns) >= len(train_feature_cols):
        X = df.iloc[:, :len(train_feature_cols)].to_numpy().astype(np.float32)
        feature_names = train_feature_cols
    else:
        # 如果列数不匹配，尝试使用所有列
        X = df.to_numpy().astype(np.float32)
        feature_names = df.columns.tolist()[:X.shape[1]]
        print(f"警告: 数据列数与训练数据不匹配，使用前 {len(feature_names)} 列")

    # 归一化待解释数据
    X_norm = input_normalization_param(X, norm_params)

    # 创建 SHAP 解释器
    explainer = create_shap_explainer(model, train_features_norm, args.nsamples)

    # 生成 SHAP 可视化
    generate_shap_plots(explainer, X_norm, feature_names, args.output_dir, args.plot_type)

    print(f"\nSHAP 分析完成！结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
