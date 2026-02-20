import torch
import sys

# 模型文件路径（默认为当前目录下的 stories42M.pt）
model_path = 'stories42M.pt'
if len(sys.argv) > 1:
    model_path = sys.argv[1]

print(f"正在检查模型文件: {model_path}")

try:
    # 加载模型文件（使用 map_location='cpu' 避免 GPU 依赖）
    model_data = torch.load(model_path, map_location='cpu')
    
    # 检查是否包含 'model' 键（通常模型权重存储在该键下）
    if 'model' in model_data:
        state_dict = model_data['model']
    elif isinstance(model_data, dict):
        # 若直接是 state_dict
        state_dict = model_data
    else:
        print('无法识别模型文件结构')
        sys.exit(1)
    
    # 遍历前几个参数，打印数据类型
    print('\n模型参数数据类型（前 5 个）：')
    for i, (name, param) in enumerate(state_dict.items()):
        if i < 5:  # 只打印前 5 个参数的类型，避免输出过多
            print(f'{name}: {param.dtype}')
        # 检查是否有非 float32 的类型
        if param.dtype != torch.float32:
            print(f'发现非 float32 类型：{name} -> {param.dtype}')
    
    # 统计所有参数的数据类型
    dtype_counts = {}
    for param in state_dict.values():
        dtype = str(param.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    print('\n所有参数数据类型统计：')
    for dtype, count in dtype_counts.items():
        print(f'{dtype}: {count}')
    
    print('\n检查完成！')
    
except Exception as e:
    print(f"检查过程中出错：{e}")
    sys.exit(1)
