# shuffle=True vs shuffle=False 的合理性
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn)
dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                            collate_fn=dev_dataset.collate_fn)

## answer 

- 训练数据加载器 ( train_dataloader ) 设置了 shuffle=True
- 验证数据加载器 ( dev_dataloader ) 设置了 shuffle=False
## 为什么设置不同？
### 1. 训练数据使用 shuffle=True 的原因
1. 打破数据顺序依赖 ：
   
   - 原始数据可能具有某种顺序（例如，按类别排序），这会导致模型在训练过程中连续看到相似的数据
   - 随机打乱数据可以确保模型不会学习到数据的顺序模式，而是专注于数据本身的特征
2. 提高模型泛化能力 ：
   
   - 打乱数据可以防止模型对特定的数据顺序产生过拟合
   - 每次迭代都以不同的顺序看到数据，有助于模型学习更一般化的模式
3. 优化梯度下降 ：
   
   - 随机梯度下降（SGD）及其变体在随机数据上表现更好
   - 打乱数据可以减少训练过程中的噪声，使梯度更新更加稳定
4. 避免局部最优 ：
   
   - 数据顺序固定可能导致模型陷入局部最优解
   - 随机打乱有助于模型探索参数空间的不同区域
### 2. 验证数据使用 shuffle=False 的原因
1. 可重现性 ：
   
   - 固定的验证数据顺序确保每次评估的结果一致
   - 便于比较不同模型或不同训练阶段的性能
2. 计算效率 ：
   
   - 验证过程不需要打乱数据，避免了不必要的计算开销
   - 对于大型验证集，打乱操作会增加计算时间
3. 稳定评估 ：
   
   - 固定的评估顺序确保评估指标的稳定性
   - 便于分析模型在特定类型数据上的表现
4. 内存使用 ：
   
   - 对于某些评估指标的计算，可能需要保持数据的原始顺序
   - 固定顺序可以更有效地使用内存
## 实际效果
### 训练过程中的影响
当使用 shuffle=True 训练时：

- 模型在每次 epoch 开始时都会看到重新打乱的数据
- 损失函数的值可能会有一定的波动，但整体趋势应该是下降的
- 模型学习到的特征更加全面，泛化能力更强
### 验证过程中的影响
当使用 shuffle=False 验证时：

- 每次评估都使用相同顺序的验证数据
- 评估指标（如准确率、F1 分数）更加稳定
- 可以更可靠地比较不同模型或不同训练阶段的性能
 
 
 
 
 
 
 
 ### 如何进一步优化加载效率 // after try, both 2 and 3 degrade the performance
 当前的实现已经是合理的，但可以考虑以下优化：

1. 使用 worker_init_fn ：
   - 对于多线程数据加载，可以设置 worker_init_fn 来确保每个工作线程都有不同的随机种子
   - 这可以进一步提高数据打乱的随机性
2. 增加 pin_memory ：
   
   - 对于 GPU 训练，可以设置 pin_memory=True 来加速数据从 CPU 到 GPU 的传输
   - 例如： DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn, pin_memory=True) 

3. 设置 num_workers ：
   
   - 根据系统 CPU 核心数量设置适当的 num_workers 值
   - 这可以并行加载数据，提高训练速度
   - 例如： DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn, num_workers=4)
                            
                            
### Test-1
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
								  collate_fn=train_dataset.collate_fn,num_workers=4)  


### 观察
 性能反而变差了
train-0: 100%|████████████████████████████████████████████████████████████████████| 428/428 [00:24<00:00, 17.77it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████████| 428/428 [00:12<00:00, 35.43it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 80.54it/s] 
save the model to finetune-10-1e-06.pt██████████████████████████████████████████    | 53/56 [00:00<00:00, 81.41it/s] 
epoch 0: train loss :: 3.686, train acc :: 0.272, dev acc :: 0.252
train-1: 100%|████████████████████████████████████████████████████████████████████| 428/428 [00:24<00:00, 17.64it/s] 
train-1: 100%|███████████████████████████████████████████████████████████████████▊| 427/428 [00:23<00:00, 25.99it/s]Traceback (most recent call last): 

## 分析
在优化数据加载时，可以监控以下指标：
GPU 利用率：如果 GPU 利用率低，可能是数据加载成为瓶颈
CPU 利用率：如果 CPU 利用率过高，可能需要减少 worker 数量
内存使用：确保内存使用在合理范围内
数据加载时间：测量每个 batch 的加载时间，找出瓶颈




### Test-2
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
								  collate_fn=train_dataset.collate_fn,num_workers=2,pin_memory=True)  


 python run_llama.py --option finetune --epochs 5 --lr 1e-6 --batch_size 20  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt --use_gpu
args: {'train': 'data/sst-train.txt', 'dev': 'data/sst-dev.txt', 'test': 'data/sst-test.txt', 'label_names': 'data/sst-label-mapping.json', 'pretrained_model_path': 'stories42M.pt', 'max_sentence_len': None, 'seed': 1337, 'epochs': 5, 'option': 'finetune', 'use_gpu': True, 'generated_sentence_low_temp_out': 'generated-sentence-temp-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-output.txt', 'test_out': 'sst-test-finetuning-output.txt', 'batch_size': 20, 'hidden_dropout_prob': 0.3, 'lr': 1e-06}
load 8544 data from data/sst-train.txt
load 1101 data from data/sst-dev.txt
-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-output.txt', 'test_out': 'sst-test-finetuning-output.txt', 'batch_size': 20, 'hidden_dropout_prob': 0.3, 'lr': 1e-06}
load 8544 data from data/sst-train.txt
-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-output.txt', 'test_out': 'sst-test-finetuning-output.txt', 'batch_size': 20, 'hidden_dropout_prob': 0.3, 'lr': 1e-06}
-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-output.txt', 'test_out': 'sst-test-finetuning-output.txt', 'batch_size': 20, 'hidden_dropout_prob': 0.3, 'lr': 1e-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-ou-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-output.txt', 'test_out': 'sst-test-finetuning-output.txt', 'batch_size': 20, 'hidden_dropout_prob': 0.3, 'lr': 1e-0.txt', 'generated_sentence_high_temp_out': 'generated-sentence-temp-1.txt', 'dev_out': 'sst-dev-finetuning-output.txt', 'test_out': 'sst-test-finetuning-output.txt', 'batch_size': 20, 'hidden_dropout_prob': 0.3, 'lr': 1e-06}
load 8544 data from data/sst-train.txt
load 1101 data from data/sst-dev.txt
C:\Users\hongf\miniconda3\envs\Build_Your_Own_LLaMa-Practice\minllama-assignment-master\llama.py:337: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint_dict = torch.load(checkpoint, map_location=device)
train-0: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:20<00:00, 20.87it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:08<00:00, 50.57it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 72.63it/s] 
save the model to finetune-5-1e-06.pt█████████████████████████████████▊       | 50/56 [00:00<00:00, 77.43it/s] 
epoch 0: train loss :: 2.089, train acc :: 0.269, dev acc :: 0.272
train-1: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:20<00:00, 20.76it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:08<00:00, 49.18it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 81.86it/s] 
save the model to finetune-5-1e-06.pt██████████████████████████████████████▌  | 54/56 [00:00<00:00, 83.37it/s] 
epoch 1: train loss :: 1.821, train acc :: 0.309, dev acc :: 0.278
train-2: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:20<00:00, 20.89it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:08<00:00, 49.70it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 83.85it/s] 
save the model to finetune-5-1e-06.pt██████████████████████████████████████▌  | 54/56 [00:00<00:00, 83.83it/s] 
epoch 2: train loss :: 1.724, train acc :: 0.317, dev acc :: 0.290
train-3: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:20<00:00, 20.93it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:09<00:00, 45.87it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 81.81it/s] 
save the model to finetune-5-1e-06.pt██████████████████████████████████████▌  | 54/56 [00:00<00:00, 81.52it/s] 
epoch 3: train loss :: 1.648, train acc :: 0.353, dev acc :: 0.333
train-4: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:20<00:00, 20.44it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:08<00:00, 49.65it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 85.82it/s]
save the model to finetune-5-1e-06.pt██████████████████████████████████████▌  | 54/56 [00:00<00:00, 86.03it/s]
epoch 4: train loss :: 1.569, train acc :: 0.378, dev acc :: 0.344



##
restore back:
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
								  collate_fn=train_dataset.collate_fn)  

## result is better than adjustment, why ??
  checkpoint_dict = torch.load(checkpoint, map_location=device)
train-0: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:17<00:00, 24.48it/s] 
  checkpoint_dict = torch.load(checkpoint, map_location=device)
train-0: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:17<00:00, 24.48it/s] 
train-0: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:17<00:00, 24.48it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:05<00:00, 81.90it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 81.72it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:05<00:00, 81.90it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 81.72it/s] 
save the model to finetune-5-1e-06.pt██████████████████████████████████████▌  | 54/56 [00:00<00:00, 82.27it/s] 
epoch 0: train loss :: 2.089, train acc :: 0.269, dev acc :: 0.272
train-1: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:17<00:00, 24.94it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:05<00:00, 82.14it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 83.54it/s] 
save the model to finetune-5-1e-06.pt█████████████████████████████████████▍   | 53/56 [00:00<00:00, 85.22it/s] 
epoch 1: train loss :: 1.821, train acc :: 0.309, dev acc :: 0.278
train-2: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:17<00:00, 25.15it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:05<00:00, 83.33it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 82.90it/s] 
save the model to finetune-5-1e-06.pt█████████████████████████████████████▍   | 53/56 [00:00<00:00, 82.79it/s] 
epoch 2: train loss :: 1.724, train acc :: 0.317, dev acc :: 0.290
train-3: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:16<00:00, 25.71it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:05<00:00, 83.37it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 82.74it/s] 
save the model to finetune-5-1e-06.pt█████████████████████████████████████▍   | 53/56 [00:00<00:00, 82.45it/s]
epoch 3: train loss :: 1.648, train acc :: 0.353, dev acc :: 0.333
train-4: 100%|██████████████████████████████████████████████████████████████| 428/428 [00:16<00:00, 25.47it/s] 
eval: 100%|█████████████████████████████████████████████████████████████████| 428/428 [00:05<00:00, 78.71it/s] 
eval: 100%|███████████████████████████████████████████████████████████████████| 56/56 [00:00<00:00, 81.67it/s] 
save the model to finetune-5-1e-06.pt█████████████████████████████████████▍   | 53/56 [00:00<00:00, 79.76it/s]
epoch 4: train loss :: 1.569, train acc :: 0.378, dev acc :: 0.344                      