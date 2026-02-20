import warnings
# Ignore pynvml deprecation warning
warnings.filterwarnings("ignore", message="The pynvml package is deprecated. Please install nvidia-ml-py instead.", category=FutureWarning)

import signal
from torch.optim.lr_scheduler import CosineAnnealingLR  #qhf enhancement 
from contextlib import nullcontext
import json
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

# 信号处理函数，用于优雅退出程序
def signal_handler(sig, frame):
    print("\n正在终止程序...")
    # 可选：添加保存模型状态、释放资源等操作
    sys.exit(0)

# 捕获 Ctrl+C 信号
signal.signal(signal.SIGINT, signal_handler)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
from classifier import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
from llama import Llama, load_pretrained
from optimizer import AdamW
from tokenizer import Tokenizer
from tqdm import tqdm
from typing import Optional


TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# create a custom Dataset Class to be used for the dataloader
class LlamaDataset(Dataset):
	def __init__(self, dataset, args, eos=False):
		self.dataset = dataset
		self.p = args
		self.tokenizer = Tokenizer(max_len=args.max_sentence_len)
		self.eos = eos

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele

	def pad_data(self, data):
		sents = [x[0] for x in data]
		labels = [x[1] for x in data]
		encoding = [self.tokenizer.encode(s, bos=True, eos=self.eos) for s in sents]
		max_length_in_batch = max([len(sentence) for sentence in encoding])
		encoding_padded = [sentence + [self.tokenizer.pad_id] * (max_length_in_batch - len(sentence)) for sentence in encoding]
		token_ids = torch.LongTensor(encoding_padded)
		labels = torch.LongTensor(labels)

		return token_ids, labels, sents

	def collate_fn(self, all_data):

		token_ids, labels, sents = self.pad_data(all_data)
		batched_data = {
				'token_ids': token_ids,
				'labels': labels,
				'sents': sents,
			}

		return batched_data

# create a custom Dataset Class for pretraining
class PretrainDataset(Dataset):
	def __init__(self, file_path, tokenizer, max_seq_len):
		self.file_path = file_path
		self.tokenizer = tokenizer
		self.max_seq_len = max_seq_len
		self.data = self._load_data()

	def _load_data(self):
		"""
		Load and preprocess pretraining data from file
		"""
		data = []
		with open(self.file_path, 'r', encoding='utf-8') as f:
			text = f.read()
			# Tokenize the entire text
			tokens = self.tokenizer.encode(text, bos=False, eos=False)
			# Split into chunks of max_seq_len
			for i in range(0, len(tokens), self.max_seq_len):
				chunk = tokens[i:i + self.max_seq_len]
				# Ensure chunk has at least 2 tokens for training
				if len(chunk) >= 2:
					data.append(chunk)
		print(f"Loaded {len(data)} chunks from {self.file_path}")
		return data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

	def collate_fn(self, batch):
		"""
		Collate function for pretraining data
		"""
		# Find the maximum length in the batch
		max_len = max(len(chunk) for chunk in batch)
		
		# Pad sequences to max_len
		padded_batch = []
		for chunk in batch:
			# Add padding to reach max_len
			padded_chunk = chunk + [self.tokenizer.pad_id] * (max_len - len(chunk))
			padded_batch.append(padded_chunk)
		
		# Convert to tensors
		token_ids = torch.LongTensor(padded_batch)
		# Create labels (shifted by one for autoregressive training)
		labels = torch.LongTensor(padded_batch)
		
		return token_ids, labels


# create the data which is a list of (sentence, label, token for the labels)
def create_data(filename, tokenizer: Tokenizer, flag: str ='train', lower: bool = False, eos: bool = True, prompt_suffix: Optional[str]=None):
	# specify the tokenizer
	num_labels = {}
	data = []

	with open(filename, 'r', encoding='utf-8') as fp:
		for line in fp:
			label, org_sent = line.split(' ||| ')
			if lower:
				org_sent = org_sent.lower()
			sent = org_sent.strip()
			if prompt_suffix is not None:
				sent = f"{sent} {prompt_suffix}"
			tokens = tokenizer.encode(sent, bos=True, eos=eos)
			label = int(label.strip())
			if label not in num_labels:
				num_labels[label] = len(num_labels)
			data.append((sent, label, tokens))
	print(f"load {len(data)} data from {filename}")
	if flag == 'train':
		return data, len(num_labels)
	else:
		return data

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(dataloader, model, device):
	model.eval() # switch to eval model, will turn off randomness like dropout
	y_true = []
	y_pred = []
	sents = []
	for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
		b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

		b_ids = b_ids.to(device)

		logits = model(b_ids)
		logits = logits.detach().cpu().numpy()
		preds = np.argmax(logits, axis=1).flatten()

		b_labels = b_labels.flatten()
		y_true.extend(b_labels)
		y_pred.extend(preds)
		sents.extend(b_sents)

	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)

	return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
	save_info = {
		'model': model.state_dict(),
		'optim': optimizer.state_dict(),
		'args': args,
		'model_config': config,
		'system_rng': random.getstate(),
		'numpy_rng': np.random.get_state(),
		'torch_rng': torch.random.get_rng_state(),
	}

	torch.save(save_info, filepath)
	print(f"save the model to {filepath}")

#qhf
def train(args):
	device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
	#### Load data
	# create the data and its corresponding datasets and dataloader
	tokenizer = Tokenizer(args.max_sentence_len)
	train_data, num_labels = create_data(args.train, tokenizer, 'train')
	dev_data = create_data(args.dev, tokenizer, 'valid')

	train_dataset = LlamaDataset(train_data, args)
	dev_dataset = LlamaDataset(dev_data, args)
    
	##qhf
	#train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
	#							  collate_fn=train_dataset.collate_fn)
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
								  collate_fn=train_dataset.collate_fn)  

	dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size,
								collate_fn=dev_dataset.collate_fn)
    #qhf false -> true


	#### Init model
	config = {'hidden_dropout_prob': args.hidden_dropout_prob,
			  'pretrained_model_path': args.pretrained_model_path,
			  'num_labels': num_labels,
			  'data_dir': '.',
			  'option': args.option}

	config = SimpleNamespace(**config)

	# initialize the Senetence Classification Model
	model = LlamaEmbeddingClassifier(config)
	model = model.to(device)
    
	#qhf
	lr = args.lr
	## specify the optimizer
	#optimizer = AdamW(model.parameters(), lr=lr)
	#qhf enhancement
	optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0005)  # 0.01->0.001

    #qhf enhancement
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=lr * 1e-3)

	best_dev_acc = 0

	## run for the specified number of epochs
	###train the model for the specified number of epochs
	
	for epoch in tqdm(range(args.epochs)):
		model.train()
		train_loss = 0
		num_batches = 0
		for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
			b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

			b_ids = b_ids.to(device)
			b_labels = b_labels.to(device)

			optimizer.zero_grad()
			logits = model(b_ids)
			loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			num_batches += 1

            #qhf enhancement
			scheduler.step()   # 每个batch更新一次学习率

		train_loss = train_loss / (num_batches)

		train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
		dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)
        
		#qhf
		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			save_model(model, optimizer, args, config, args.filepath)

		print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")

def generate_sentence(args, prefix, outfile, max_new_tokens = 75, temperature = 0.0):
	with torch.no_grad():
		cuda_available = args.use_gpu and torch.cuda.is_available()
		device = torch.device('cuda') if cuda_available else torch.device('cpu')
		ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32) if cuda_available else nullcontext()
		llama = load_pretrained(args.pretrained_model_path)
		llama = llama.to(device)
		print(f"load model from {args.pretrained_model_path}")
		enc = Tokenizer(args.max_sentence_len)

		start_ids = enc.encode(prefix, bos=True, eos=False)
		x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

		# run generation
		with torch.no_grad():
			with ctx:
				y = llama.generate(x, max_new_tokens, temperature=temperature)
				sentence = enc.decode(y[0].tolist())
				print(f"Temperature is {temperature}")
				print(sentence)
				print('---------------')
				writer = open(outfile, 'w', encoding='utf-8')
				writer.write(sentence)
				print(f"Wrote generated sentence to {outfile}.")
				writer.close()

def write_predictions_to_file(split: str, outfile: str, acc: float, pred: list[str], sents: list[str]):
	with open(outfile, "w+", encoding='utf-8') as f:
		print(f"{split} acc :: {acc :.3f}")
		for s, p in zip(sents, pred):
			f.write(f"{p} ||| {s}\n")

def test_with_prompting(args):
	assert args.dev_out.endswith("dev-prompting-output.txt"), 'For saving prompting results, please set the dev_out argument as "<dataset>-dev-prompting-output.txt"'
	assert args.test_out.endswith("test-prompting-output.txt"), 'For saving prompting results, please set the test_out argument as "<dataset>-test-prompting-output.txt"'

	with torch.no_grad():

		device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
		#### Load data
		# create the data and its corresponding datasets and dataloader
		tokenizer = Tokenizer(args.max_sentence_len)
		label_names = json.load(open(args.label_names, 'r', encoding='utf-8'))
		_, num_labels = create_data(args.train, tokenizer, 'train')

		#### Init model
		config = {'pretrained_model_path': args.pretrained_model_path,
				'label_names': label_names,
				'num_labels': num_labels,
				'data_dir': '.',
				'option': args.option}

		config = SimpleNamespace(**config)

		if len(label_names) == 2:
			label_name_str = " or ".join(label_names)
		else:
			label_name_str = ", ".join(label_names[:-1]) + ", or " + label_names[-1]
		prompt_suffix=f"Is this movie {label_name_str}? This movie is "
		model = LlamaZeroShotClassifier(config, tokenizer, label_names)
		model = model.to(device)

		dev_data = create_data(args.dev, tokenizer, 'valid', eos=False, prompt_suffix=prompt_suffix)
		dev_dataset = LlamaDataset(dev_data, args, eos=False)
		dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)
		#qhf false ->true

		test_data = create_data(args.test, tokenizer, 'test', eos=False, prompt_suffix=prompt_suffix)
		test_dataset = LlamaDataset(test_data, args, eos=False)
		test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
		#qhf false ->true

		dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
		test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

		write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
		write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)

def test(args):
	assert args.dev_out.endswith("dev-finetuning-output.txt"), 'For saving finetuning results, please set the dev_out argument as "<dataset>-dev-finetuning-output.txt"'
	assert args.test_out.endswith("test-finetuning-output.txt"), 'For saving finetuning results, please set the test_out argument as "<dataset>-test-finetuning-output.txt"'
	with torch.no_grad():
		device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
		saved = torch.load(args.filepath)
		config = saved['model_config']
		model = LlamaEmbeddingClassifier(config)
		model.load_state_dict(saved['model'])
		model = model.to(device)
		print(f"load model from {args.filepath}")
		tokenizer = Tokenizer(args.max_sentence_len)
		dev_data = create_data(args.dev, tokenizer, 'valid')
		dev_dataset = LlamaDataset(dev_data, args)
		dev_dataloader = DataLoader(dev_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)
        #qhf false ->true




	test_data = create_data(args.test, tokenizer, 'test')
	test_dataset = LlamaDataset(test_data, args)
	test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        #qhf false ->true

	dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
	test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

	write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
	write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)

# pretrain function for training from scratch
def pretrain(args):
	"""
	Pretrain a Llama model from scratch
	"""
	# Set up device
	device = torch.device('cuda') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')
	print(f"Using device: {device}")

	# Create save directory if it doesn't exist
	os.makedirs(args.save_dir, exist_ok=True)

	# Initialize tokenizer
	tokenizer = Tokenizer()

	# Load pretraining data
	dataset = PretrainDataset(args.pretrain_data, tokenizer, args.max_seq_len)
	dataloader = DataLoader(
		dataset, 
		batch_size=args.pretrain_batch_size, 
		shuffle=True, 
		collate_fn=dataset.collate_fn,
		num_workers=4
	)

	# Initialize model
	from config import LlamaConfig
	model_config = LlamaConfig(
		vocab_size=tokenizer.n_words,
		dim=512,
	
dropout=0.1,
		n_layers=8,
		n_heads=8,
		n_kv_heads=8,
		max_seq_len=args.max_seq_len,
		layer_norm_eps=1e-5,
		multiple_of=32,
		hidden_dim=None,
		position_embedding_type="rotary",
		use_cache=True
	)

	model = Llama(model_config)
	model = model.to(device)

	# Set up optimizer
	from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR
	optimizer = AdamW(
		model.parameters(), 
		lr=args.pretrain_lr, 
		weight_decay=args.weight_decay
	)

	# Calculate total steps
	total_steps = args.epochs * len(dataloader) // args.gradient_accumulation_steps

	# Calculate warmup steps based on ratio if needed
	warmup_steps = args.warmup_steps
	if warmup_steps == 0 and args.warmup_ratio > 0:
		warmup_steps = int(total_steps * args.warmup_ratio)

	# Set up learning rate scheduler with warmup
	warmup_scheduler = LinearLR(
		optimizer, 
		start_factor=0.01, 
		end_factor=1.0, 
		total_iters=warmup_steps
	)

	# Set up main learning rate scheduler
	if args.lr_scheduler == "cosine":
		main_scheduler = CosineAnnealingLR(
			optimizer, 
			T_max=total_steps - warmup_steps,
			eta_min=args.min_lr
		)
	elif args.lr_scheduler == "linear":
		from torch.optim.lr_scheduler import LinearLR as LinearDecayLR
		main_scheduler = LinearDecayLR(
			optimizer, 
			start_factor=1.0, 
			end_factor=args.min_lr / args.pretrain_lr, 
			total_iters=total_steps - warmup_steps
		)
	else:  # constant
		main_scheduler = ConstantLR(
			optimizer, 
			factor=1.0, 
			total_iters=total_steps - warmup_steps
		)

	# Mixed precision training
	if args.use_gpu:
		scaler = torch.amp.GradScaler('cuda')
	else:
		scaler = torch.amp.GradScaler('cpu')

	# Training loop
	global_step = 0
	best_loss = float('inf')

	for epoch in range(args.epochs):
		model.train()
		train_loss = 0
		num_batches = 0

		for step, (token_ids, labels) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')):
			# Move data to device
			token_ids = token_ids.to(device)
			labels = labels.to(device)

			# Zero gradients
			if step % args.gradient_accumulation_steps == 0:
				optimizer.zero_grad()

			# Forward pass with mixed precision
			if args.use_gpu:
				with torch.amp.autocast('cuda'):
					# Get model outputs
					logits, _ = model(token_ids, targets=labels)
					
					# Calculate loss
					# Shift labels for autoregressive training
					shifted_logits = logits[..., :-1, :].contiguous()
					shifted_labels = labels[..., 1:].contiguous()
					
					# Flatten the tensors
					loss = F.cross_entropy(
						shifted_logits.view(-1, shifted_logits.size(-1)),
						shifted_labels.view(-1),
						ignore_index=tokenizer.pad_id
					)
					
					# Scale loss for gradient accumulation
					loss = loss / args.gradient_accumulation_steps
			else:
				# Get model outputs
				logits, _ = model(token_ids, targets=labels)
				
				# Calculate loss
				# Shift labels for autoregressive training
				shifted_logits = logits[..., :-1, :].contiguous()
				shifted_labels = labels[..., 1:].contiguous()
				
				# Flatten the tensors
				loss = F.cross_entropy(
					shifted_logits.view(-1, shifted_logits.size(-1)),
					shifted_labels.view(-1),
					ignore_index=tokenizer.pad_id
				)
				
				# Scale loss for gradient accumulation
				loss = loss / args.gradient_accumulation_steps

			# Backward pass
			scaler.scale(loss).backward()

			# Update parameters
			if (step + 1) % args.gradient_accumulation_steps == 0:
				# Gradient clipping
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

				# Update parameters
				scaler.step(optimizer)
				scaler.update()

				# Update learning rate
				if global_step < warmup_steps:
					warmup_scheduler.step()
				else:
					main_scheduler.step()

				# Accumulate loss
				train_loss += loss.item() * args.gradient_accumulation_steps
				num_batches += 1
				global_step += 1

				# Save checkpoint
				if global_step % args.checkpoint_interval == 0:
					checkpoint_path = os.path.join(args.save_dir, f'checkpoint-step-{global_step}.pt')
					save_model(model, optimizer, args, model_config, checkpoint_path)

		# Calculate average loss for epoch
		if num_batches > 0:
			avg_loss = train_loss / num_batches
			current_lr = optimizer.param_groups[0]['lr']

			print(f"Epoch {epoch+1}/{args.epochs}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

			# Save best model at specified interval
			if (epoch + 1) % args.best_model_interval == 0:
				if avg_loss < best_loss:
					best_loss = avg_loss
					best_model_path = os.path.join(args.save_dir, f'best-model-epoch-{epoch+1}.pt')
					save_model(model, optimizer, args, model_config, best_model_path)
			
			# Save model at specified interval
			if (epoch + 1) % args.save_interval == 0:
				interval_model_path = os.path.join(args.save_dir, f'model-epoch-{epoch+1}.pt')
				save_model(model, optimizer, args, model_config, interval_model_path)
		else:
			print(f"Epoch {epoch+1}/{args.epochs}: No batches processed (insufficient data for gradient accumulation)")

	# Save final model
	final_model_path = os.path.join(args.save_dir, 'final-model.pt')
	save_model(model, optimizer, args, model_config, final_model_path)
	print(f"Pretraining completed. Final model saved to {final_model_path}")

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
	parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
	parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
	parser.add_argument("--label-names", type=str, default="data/cfimdb-label-mapping.json")
	parser.add_argument("--pretrained-model-path", type=str, default="stories42M.pt")
	parser.add_argument("--max_sentence_len", type=int, default=None)
	parser.add_argument("--seed", type=int, default=1337)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--option", type=str,
					help='prompt: the Llama parameters are frozen; finetune: Llama parameters are updated; pretrain: train from scratch',
					choices=('generate', 'prompt', 'finetune', 'pretrain'), default="generate")
	parser.add_argument("--use_gpu", action='store_true')
	parser.add_argument("--generated_sentence_low_temp_out", type=str, default="generated-sentence-temp-0.txt")
	parser.add_argument("--generated_sentence_high_temp_out", type=str, default="generated-sentence-temp-1.txt")
	parser.add_argument("--dev_out", type=str, default="cfimdb-dev-prompting-output.txt")
	parser.add_argument("--test_out", type=str, default="cfimdb-test-prompting-output.txt")

	# hyper parameters
	parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
	parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)  # qhf 0.3->-0.8
	parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
					default=2e-5)

	# pretrain specific parameters
	parser.add_argument("--pretrain_data", type=str, default="./data/pretrain_corpus.txt", help="Path to pretraining corpus")
	parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length for pretraining")
	parser.add_argument("--pretrain_batch_size", type=int, default=8, help="Batch size for pretraining")
	parser.add_argument("--pretrain_lr", type=float, default=1e-4, help="Learning rate for pretraining")
	parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps for learning rate") ## qhf
	parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Ratio of warmup steps to total steps")
	parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum learning rate after annealing")
	parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="Learning rate scheduler type")
	parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
	parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Interval to save checkpoints")  ## 
	parser.add_argument("--save_interval", type=int, default=50, help="Interval (in epochs) to save models")
	parser.add_argument("--best_model_interval", type=int, default=20, help="Interval (in epochs) to check and save best model")  # every 20 epochs save 
	parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

	args = parser.parse_args()
	print(f"args: {vars(args)}")
	return args

if __name__ == "__main__":
	args = get_args()
	args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt' # save path
	seed_everything(args.seed)  # fix the seed for reproducibility

	if args.option == "generate":
		# Step 1
		# Complete this sentence to test your implementation!
		#prefix = "I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"
		#prefix = "中国的四大名著,有一本是关于孙悟空的,今天要讲的故事是三打白"
		prefix = "I want to hear the sotory about the Mickey Mouse, the sotry is"
		generate_sentence(args, prefix, args.generated_sentence_low_temp_out, max_new_tokens=75, temperature=0.0)
		generate_sentence(args, prefix, args.generated_sentence_high_temp_out, max_new_tokens=75, temperature=1.0)
	elif args.option == "prompt":
		# Step 2
		# Solve this task with prompted language modeling
		test_with_prompting(args)
	elif args.option == "finetune":
		# Step 3
		# Finetune a classification model
		train(args)

		# Step 4
		# Evaluate your model on the dev and test sets
		test(args)
	elif args.option == "pretrain":
		# Step 5
		# Pretrain a model from scratch
		pretrain(args)
	else:
		raise ValueError(f"Invalid option: {args.option}")