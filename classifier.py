
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		# 存储类别数量
		self.num_labels = config.num_labels
		# 加载预训练的LLaMA模型
		# 关键设计：使用预训练模型的语言理解能力进行零样本分类
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		# 核心设计：冻结LLaMA模型参数，因为零样本分类不需要微调
		# 这样可以减少计算量，同时保持模型的预训练能力
		for param in self.llama.parameters():
			param.requires_grad = False   ###
		# 验证标签数量与配置一致
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		# 将标签名称编码为token IDs，去除bos和eos标记
		# 这样可以直接计算模型生成这些标签的概率


		#使用列表推导式，为每个标签名称创建对应的token ID序列，并存储在 self.label_name_ids 中
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]
		##
		#在 create_data 函数中,  分割标签和文本, 结果: label = "3", org_sent = " But he somehow pulls it off
		#去除首尾空白,结果: sent = "But he somehow pulls it off ."
		# 添加 prompt 后缀（零样本分类时使用）
		#prompt_suffix=f"Is this movie {label_name_str}? This movie is "
		#处理后的文本变为: "But he somehow pulls it off . Is this movie positive or negative? This movie is "

		#tokenizer.encode(sent, bos=True, eos=eos)  # 编码文本
		#后续 ,label = int(label.strip())  # 将标签转换为整数 # 结果: label = 3



	def forward(self, input_ids):
		# compute the completion probability of each label string
		# 核心步骤1：获取LLaMA模型的输出logits
		# logits形状: (batch_size, seq_len, vocab_size)
		logits, _ = self.llama(input_ids)
		#### self.llama(input_ids) 的返回值
		#根据代码上下文， self.llama(input_ids) 返回两个值：
		#- 第一个返回值 ： logits - 模型的原始输出分数，形状为 (batch_size, seq_len, vocab_size)
		#batch_size 是指模型一次处理的样本数量，是深度学习中的重要超参数
        #- 第二个返回值 ：隐藏状态 h - 模型中间层的输出，形状为 (batch_size, seq_len, hidden_dim)

		# 核心步骤2：计算每个位置的log概率分布
		# 将logits转换为概率分布，使用log_softmax数值更稳定
		log_probabilities = F.log_softmax(logits, dim=-1)  ##函数对模型输出的 logits 进行处理，计算对数概率分布
		# 初始化标签概率矩阵
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		# 核心步骤3：计算每个标签的总概率
		# 对每个标签，计算模型生成该标签所有token的概率之和
		for i, label_token_ids in enumerate(self.label_name_ids):
			# 计算每个位置生成标签token的概率之和
			#[:, :, label_token_ids] : 在词汇表维度上，只选择 label_token_ids 对应的位置
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			# 取第一个位置的概率作为该标签的概率
			label_probabilities[:, i] = total_log_prob[:, 0]
		# 返回各标签的概率
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		# 存储类别数量
		self.num_labels = config.num_labels
		# 加载预训练的LLaMA模型
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		# 核心设计：根据训练模式决定是否冻结LLaMA参数
		# pretrain模式：冻结LLaMA，仅训练分类头，适合数据量小的情况
		# finetune模式：微调整个模型，适合数据量足够的情况
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		# 添加dropout层，防止过拟合
		# 关键设计：在分类头前添加dropout，提高模型泛化能力
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		# 添加线性分类头
		# 核心设计：将LLaMA的隐藏状态映射到类别空间
		# 输入维度：LLaMA的隐藏层维度
		# 输出维度：类别数量
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# Get the hidden states from the llama model
		# 核心步骤1：获取LLaMA模型的输出
		# 这里使用下划线忽略logits，只保留隐藏状态h
		_, h = self.llama(input_ids)
		# Find the hidden state after the final token of each input sequence
		# Assuming input_ids shape is (batch_size, seq_len), we take the last token's hidden state
		# 核心步骤2：提取每个序列最后一个token的隐藏状态
		# 关键设计：使用最后一个token的隐藏状态作为整个序列的表示
		# 这是一种常见的做法，假设最后一个token包含了整个序列的语义信息
		last_hidden_state = h[:, -1, :]
		# Apply dropout
		# 核心步骤3：应用dropout防止过拟合
		last_hidden_state = self.dropout(last_hidden_state)
		# Pass through the classifier head to get logits
		# 核心步骤4：通过分类头计算logits
		# 将隐藏状态映射到类别空间
		logits = self.classifier_head(last_hidden_state)
		#维度转换 ：将高维隐藏状态（维度为 self.llama.config.dim ）转换为类别空间（维度为 self.num_labels ）
		#生成预测基础 ：计算出的 logits 是后续计算概率分布的基础
		#其核心目的是将 LLaMA 模型学习到的序列语义表示转换为分类任务所需的类别得分。
		#这一行代码体现了如何将预训练语言模型的强大语义理解能力应用到具体的分类任务中，
		#是连接特征提取和分类预测的桥梁。


		# Take log-softmax to get log-probabilities
		# 核心步骤5：计算log概率分布
		# 使用log_softmax数值更稳定，适合计算交叉熵损失
		log_probabilities = F.log_softmax(logits, dim=-1)
		# 返回各标签的概率
		return log_probabilities