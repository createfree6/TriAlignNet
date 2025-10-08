from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer,LlamaForCausalLM
import os
import time
import warnings
import numpy as np

def norm(input_emb):
    input_emb=input_emb- input_emb.mean(1, keepdim=True).detach()
    input_emb=input_emb/torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5)
   
    return input_emb
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)  
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  
                x = F.relu(x)
                x = self.dropout(x)  
        return x
warnings.filterwarnings('ignore')





class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        args.task_name = 'long_term_forecast'
        super(Exp_Long_Term_Forecast, self).__init__(args)
        configs=args
        self.text_path=configs.text_path
        self.prompt_weight=configs.prompt_weight
        self.attribute="final_sum"
        self.type_tag=configs.type_tag
        self.text_len=configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len=configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type=configs.pool_type
        self.use_fullmodel=configs.use_fullmodel
        self.hug_token=configs.huggingface_token
        mlp_sizes=[self.d_llm,int(self.d_llm/8),self.text_embedding_dim]
        self.Doc2Vec=False
        if mlp_sizes is not None:
            # self.mlp = MLP(mlp_sizes,dropout_rate=0.3)
            self.mlp = nn.Sequential(
                nn.Linear(mlp_sizes[0], mlp_sizes[1]),
                nn.ReLU(),
                nn.Linear(mlp_sizes[1], mlp_sizes[2]),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            # print number of parameters of self.model
            num_params = sum(p.numel() for p in self.model.parameters())
            print(f'Number of parameters in TS model: {num_params}')
            # print number of parameters of self.mlp
            num_params_mlp = sum(p.numel() for p in self.mlp.parameters())
            print(f'Number of parameters in MLP: {num_params_mlp}')
            print(f'Total number of parameters: {num_params + num_params_mlp}')
        else:
            self.mlp = None
        mlp_sizes2=[self.text_embedding_dim+self.args.pred_len,self.args.pred_len]
        if mlp_sizes2 is not None:
            self.mlp_proj = MLP(mlp_sizes2,dropout_rate=0.3)

        self.language_to_time_series_projection = nn.Sequential(
            nn.Linear(self.d_llm, 12),
            nn.ReLU()
        ).cuda()

        if configs.llm_model == 'Doc2Vec':
            print('Cannot using Doc2Vec')
            print("Training Doc2Vec model")
            raise Exception('Doc2Vec model is not supported')
        else:
            if configs.llm_model == 'LLAMA2':
                import os
                # 本地模型路径（你下载好的 LLaMA-7B）
                local_model_path = r"E:\code\multimodal_TSF\huggingface_cache\hub\models--huggyllama--llama-7b\snapshots"
                # HuggingFace Hub 模型名
                hf_model_name = "huggyllama/llama-7b"

                # 优先使用本地路径加载 config
                if os.path.exists(local_model_path):
                    print(f"✅ 使用本地模型路径: {local_model_path}")
                    self.llama_config = LlamaConfig.from_pretrained(local_model_path)
                else:
                    print(f"⚠️ 本地路径 {local_model_path} 未找到，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.llama_config = LlamaConfig.from_pretrained(hf_model_name, cache_dir="./huggingface_cache")

                # 修改配置
                self.llama_config.num_hidden_layers = configs.llm_layers
                self.llama_config.output_attentions = True
                self.llama_config.output_hidden_states = True

                # 加载模型
                try:
                    if os.path.exists(local_model_path):
                        self.llm_model = LlamaModel.from_pretrained(
                            local_model_path,
                            trust_remote_code=True,
                            local_files_only=True,
                            config=self.llama_config,
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        hf_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        cache_dir="./huggingface_cache"
                    )

                # 加载 tokenizer
                try:
                    if os.path.exists(local_model_path):
                        self.tokenizer = LlamaTokenizer.from_pretrained(
                            local_model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local tokenizer files not found. Attempting to download...")
                    self.tokenizer = LlamaTokenizer.from_pretrained(
                        hf_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )

            elif configs.llm_model == 'CLIP':
                import os
                from transformers import CLIPModel, CLIPProcessor, CLIPConfig

                # 根目录（snapshots 文件夹的上一层）
                root_dir = r"E:\code\multimodal_TSF\huggingface_cache\hub\models--openai--clip-vit-base-patch32\snapshots"
                hf_model_name = "openai/clip-vit-base-patch32"

                # 自动查找包含 pytorch_model.bin 和 config.json 的目录
                local_model_path = None
                for root, dirs, files in os.walk(root_dir):
                    if "pytorch_model.bin" in files and "config.json" in files:
                        local_model_path = root
                        print(f"✅ 找到本地 CLIP 模型文件夹: {local_model_path}")
                        break

                # 加载模型
                try:
                    if local_model_path:
                        # 显式加载 config
                        config_path = os.path.join(local_model_path, "config.json")
                        clip_config = CLIPConfig.from_pretrained(config_path)

                        # 模型
                        self.llm_model = CLIPModel.from_pretrained(
                            local_model_path,
                            config=clip_config,
                            local_files_only=True
                        )

                        # Processor
                        self.processor = CLIPProcessor.from_pretrained(
                            local_model_path,
                            local_files_only=True
                        )
                    else:
                        raise FileNotFoundError("本地 CLIP 模型未找到")
                except Exception as e:
                    print(f"⚠️ 本地模型加载失败 ({e})，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.llm_model = CLIPModel.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )
                    self.processor = CLIPProcessor.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )

                # ✅ 自动绑定 tokenizer
                self.tokenizer = self.processor.tokenizer


            elif configs.llm_model == 'RoBERTa':
                import os
                from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

                # 本地模型路径（你下载好的 RoBERTa-base）
                local_model_path = r"E:\code\multimodal_TSF\huggingface_cache\hub\models--roberta-base\snapshots"
                # HuggingFace Hub 模型名
                hf_model_name = "roberta-base"

                # 优先使用本地路径
                if os.path.exists(local_model_path):
                    print(f"✅ 使用本地模型路径: {local_model_path}")
                    self.roberta_config = RobertaConfig.from_pretrained(local_model_path)
                else:
                    print(f"⚠️ 本地路径 {local_model_path} 未找到，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.roberta_config = RobertaConfig.from_pretrained(hf_model_name, cache_dir="./huggingface_cache")

                # 通用配置修改
                self.roberta_config.num_hidden_layers = configs.llm_layers
                self.roberta_config.output_attentions = True
                self.roberta_config.output_hidden_states = True

                # 加载模型
                try:
                    if os.path.exists(local_model_path):
                        self.llm_model = RobertaModel.from_pretrained(
                            local_model_path,
                            local_files_only=True,
                            config=self.roberta_config,
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = RobertaModel.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        config=self.roberta_config,
                        cache_dir="./huggingface_cache"
                    )

                # 加载 tokenizer
                try:
                    if os.path.exists(local_model_path):
                        self.tokenizer = RobertaTokenizer.from_pretrained(
                            local_model_path,
                            local_files_only=True
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local tokenizer files not found. Attempting to download...")
                    self.tokenizer = RobertaTokenizer.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )

            elif configs.llm_model == 'DeepSeek':
                import os
                import torch
                from transformers import AutoTokenizer, AutoConfig, AutoModel
                from safetensors.torch import load_file as load_safetensors

                # 本地模型路径（你下载好的 DeepSeek）
                local_model_path = r"E:\code\multimodal_TSF\huggingface_cache\hub\models--deepseek\snapshots"
                # HuggingFace Hub 模型名
                hf_model_name = "deepseek-ai/DeepSeek-V3.1-Base"

                # 优先使用本地路径
                if os.path.exists(local_model_path):
                    print(f"✅ 使用本地模型路径: {local_model_path}")
                    self.deepseek_config = AutoConfig.from_pretrained(local_model_path)
                else:
                    print(f"⚠️ 本地路径 {local_model_path} 未找到，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.deepseek_config = AutoConfig.from_pretrained(hf_model_name, cache_dir="./huggingface_cache")

                # 通用配置修改（如果需要可修改）
                self.deepseek_config.output_attentions = True
                self.deepseek_config.output_hidden_states = True

                # 加载模型（优先 .safetensors）
                try:
                    safetensors_path = os.path.join(local_model_path, "model.safetensors")
                    bin_path = os.path.join(local_model_path, "pytorch_model.bin")

                    if os.path.exists(safetensors_path):
                        print("✅ 检测到 safetensors 格式，使用 from_pretrained 高效加载...")
                        self.llm_model = AutoModel.from_pretrained(
                            local_model_path,
                            local_files_only=True,
                            torch_dtype="float16",  # 强制用 FP16
                            device_map="auto",
                            trust_remote_code=True  # DeepSeek 官方代码里可能要这个
                        )
                    elif os.path.exists(bin_path):
                        print("✅ 检测到 pytorch_model.bin 格式，使用 from_pretrained 加载...")
                        self.llm_model = AutoModel.from_pretrained(
                            local_model_path,
                            local_files_only=True,
                            torch_dtype="auto",
                            device_map="auto"
                        )
                    else:
                        raise FileNotFoundError("未找到模型文件：model.safetensors 或 pytorch_model.bin")
                except Exception as e:
                    print(f"⚠️ 本地模型加载失败 ({e})，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.llm_model = AutoModel.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        config=self.deepseek_config,
                        cache_dir="./huggingface_cache"
                    )

                # 加载 tokenizer
                try:
                    if os.path.exists(local_model_path):
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            local_model_path,
                            local_files_only=True
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local tokenizer files not found. Attempting to download...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )

            elif configs.llm_model == 'Qwen2.5':
                import os
                import torch
                from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
                from safetensors.torch import load_file as load_safetensors

                # 本地模型路径（你下载好的 Qwen2.5-VL-7B-Instruct）
                local_model_path = r"E:\code\multimodal_TSF\huggingface_cache\hub\Qwen2.5-7B-Instruct\snapshots"
                # HuggingFace Hub 模型名
                hf_model_name = "Qwen/Qwen2.5-7B-Instruct"

                # 优先使用本地路径加载配置
                if os.path.exists(local_model_path):
                    print(f"✅ 使用本地模型路径: {local_model_path}")
                    self.qwen_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                else:
                    print(f"⚠️ 本地路径 {local_model_path} 未找到，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.qwen_config = AutoConfig.from_pretrained(hf_model_name, cache_dir="./huggingface_cache", trust_remote_code=True)

                # 通用配置修改（如需调试可改）
                self.qwen_config.output_attentions = True
                self.qwen_config.output_hidden_states = True

                # 加载模型（优先 .safetensors）
                try:
                    safetensors_path = os.path.join(local_model_path, "model.safetensors")
                    bin_path = os.path.join(local_model_path, "pytorch_model.bin")

                    if os.path.exists(safetensors_path):
                        print("✅ 检测到 safetensors 格式，使用 from_pretrained 高效加载...")
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            local_model_path,
                            local_files_only=True,
                            # torch_dtype=torch.float16,   # 推荐 FP16
                            # device_map="auto",
                            trust_remote_code=True
                        )
                    elif os.path.exists(bin_path):
                        print("✅ 检测到 pytorch_model.bin 格式，使用 from_pretrained 加载...")
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            local_model_path,
                            local_files_only=True,
                            config=self.qwen_config,
                            # torch_dtype="auto",
                            # device_map="auto",
                            trust_remote_code=True
                        )
                    else:
                        raise FileNotFoundError("未找到模型文件：model.safetensors 或 pytorch_model.bin")
                except Exception as e:
                    print(f"⚠️ 本地模型加载失败 ({e})，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        config=self.qwen_config,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                        cache_dir="./huggingface_cache"
                    )

                # 加载 tokenizer
                try:
                    if os.path.exists(local_model_path):
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            local_model_path,
                            local_files_only=True,
                            trust_remote_code=True
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local tokenizer files not found. Attempting to download...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        hf_model_name,
                        local_files_only=False,
                        cache_dir="./huggingface_cache",
                        trust_remote_code=True
                    )

            elif configs.llm_model == 'GPT2':
                import os
                # 本地模型路径（你下载好的 GPT2）
                local_model_path = r"E:\code\multimodal_TSF\huggingface_cache\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
                # HuggingFace Hub 模型名
                hf_model_name = "openai-community/gpt2"

                # 优先使用本地路径
                if os.path.exists(local_model_path):
                    print(f"✅ 使用本地模型路径: {local_model_path}")
                    self.gpt2_config = GPT2Config.from_pretrained(local_model_path)
                else:
                    print(f"⚠️ 本地路径 {local_model_path} 未找到，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.gpt2_config = GPT2Config.from_pretrained(hf_model_name, cache_dir="./huggingface_cache")

                # 通用配置修改
                self.gpt2_config.num_hidden_layers = configs.llm_layers
                self.gpt2_config.output_attentions = True
                self.gpt2_config.output_hidden_states = True

                # 加载模型
                try:
                    if os.path.exists(local_model_path):
                        self.llm_model = GPT2Model.from_pretrained(
                            local_model_path,
                            trust_remote_code=True,
                            local_files_only=True,
                            config=self.gpt2_config,
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = GPT2Model.from_pretrained(
                        hf_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.gpt2_config,
                        cache_dir="./huggingface_cache"
                    )

                # 加载 tokenizer
                try:
                    if os.path.exists(local_model_path):
                        self.tokenizer = GPT2Tokenizer.from_pretrained(
                            local_model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local tokenizer files not found. Attempting to download...")
                    self.tokenizer = GPT2Tokenizer.from_pretrained(
                        hf_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )

            elif configs.llm_model == 'BERT':
                import os
                # 本地模型路径（你下载好的 BERT）
                local_model_path = r"E:\code\multimodal_TSF\huggingface_cache\hub\models--google-bert--bert-base-cased\snapshots\cd5ef92a9fb2f889e972770a36d4ed042daf221e"
                # HuggingFace Hub 模型名
                hf_model_name = "google-bert/bert-base-uncased"

                # 优先使用本地路径
                if os.path.exists(local_model_path):
                    print(f"✅ 使用本地模型路径: {local_model_path}")
                    self.bert_config = BertConfig.from_pretrained(local_model_path)
                else:
                    print(f"⚠️ 本地路径 {local_model_path} 未找到，尝试从 HuggingFace 下载 {hf_model_name} ...")
                    self.bert_config = BertConfig.from_pretrained(hf_model_name, cache_dir="./huggingface_cache")

                # 通用配置修改
                self.bert_config.num_hidden_layers = configs.llm_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True

                # 加载模型
                try:
                    if os.path.exists(local_model_path):
                        self.llm_model = BertModel.from_pretrained(
                            local_model_path,
                            trust_remote_code=True,
                            local_files_only=True,
                            config=self.bert_config,
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = BertModel.from_pretrained(
                        hf_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.bert_config,
                        cache_dir="./huggingface_cache"
                    )

                # 加载 tokenizer
                try:
                    if os.path.exists(local_model_path):
                        self.tokenizer = BertTokenizer.from_pretrained(
                            local_model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    else:
                        raise FileNotFoundError
                except:
                    print("Local tokenizer files not found. Attempting to download...")
                    self.tokenizer = BertTokenizer.from_pretrained(
                        hf_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        cache_dir="./huggingface_cache"
                    )

            else:
                raise Exception('LLM model is not defined')

            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model=self.llm_model.to(self.device)
        if args.init_method == 'uniform':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == 'normal':
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError('Unsupported initialization method')
        
        self.mlp=self.mlp.to(self.device)
        self.mlp_proj=self.mlp_proj.to(self.device)
        self.learning_rate2=1e-2
        self.learning_rate3=1e-3
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.llm_model, self.tokenizer)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.998), weight_decay=0.01)
        return model_optim
    def _select_optimizer_mlp(self):
        model_optim = optim.Adam(self.mlp.parameters(), lr=self.args.learning_rate2)
        return model_optim
    def _select_optimizer_proj(self):
        model_optim = optim.Adam(self.mlp_proj.parameters(), lr=self.args.learning_rate3)
        return model_optim
    def _select_optimizer_weight(self):
        model_optim = optim.Adam([{'params': self.weight1.parameters()},
                              {'params': self.weight2.parameters()}], lr=self.args.learning_rate_weight)
        return model_optim
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, all_metric=False):
        total_loss = []
        if all_metric:
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            total_mspe = []
        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                prior_y=torch.from_numpy(vali_data.get_prior_y(index)).float().to(self.device)
                
                batch_text_embeddings = vali_data.get_text_embeddings(index)

                prompt_emb = self.mlp(batch_text_embeddings)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # batch_x is [bsz, seq_len, num_vars], prompt_emb is [bsz, seq_len, text_embedding_dim]. concatenate them in the last dimension
                batch_x = torch.cat([batch_x, prompt_emb], dim=-1)

                # dec_inp is [bsz, label_len + pred_len, num_vars], where only label_len is the true data, the rest is 0
                text_dec_inp = torch.zeros((self.args.batch_size, self.args.pred_len, self.text_embedding_dim)).to(self.device)
                text_dec_inp = torch.cat([prompt_emb[:, :self.args.label_len, :], text_dec_inp], dim=1).float().to(self.device)
                dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 1)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # TODO: this only works for single variate time series
                outputs = outputs[:, :, 0].unsqueeze(-1)
                # outputs = (1-self.prompt_weight)*outputs+self.prompt_weight*prior_y

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                if all_metric:
                    mae, mse, rmse, mape, mspe = metric(np.array(pred), np.array(true))
                    total_mae.append(mae)
                    total_mse.append(mse)
                    total_rmse.append(rmse)
                    total_mape.append(mape)
                    total_mspe.append(mspe)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        self.mlp.train()
        self.mlp_proj.train()
        if all_metric:
            total_mae = np.average(total_mae)
            total_mse = np.average(total_mse)
            total_rmse = np.average(total_rmse)
            total_mape = np.average(total_mape)
            total_mspe = np.average(total_mspe)
            return total_loss, total_mae, total_mse, total_rmse, total_mape, total_mspe
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        model_optim_mlp = self._select_optimizer_mlp()
        model_optim_proj = self._select_optimizer_proj()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.mlp.train()
            self.mlp_proj.train()
            epoch_time = time.time()
            # index: batch_size  , batch_x: batch_size seq_len OT
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                model_optim_mlp.zero_grad()
                model_optim_proj.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                prior_y=torch.from_numpy(train_data.get_prior_y(index)).float().to(self.device)  # prior_y: batch_size seq_len OT
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_text_embeddings = train_data.get_text_embeddings(index)  # 32 96 768

                prompt_emb = self.mlp(batch_text_embeddings) # 32 96 12

                # prompt_emb = prompt_emb + torch.randn_like(prompt_emb) * 50

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # 32 144 1

                # batch_x is [bsz, seq_len, num_vars], prompt_emb is [bsz, seq_len, text_embedding_dim]. concatenate them in the last dimension
                batch_x = torch.cat([batch_x, prompt_emb], dim=-1).detach()

                # dec_inp is [bsz, label_len + pred_len, num_vars], where only label_len is the true data, the rest is 0
                text_dec_inp = torch.zeros((self.args.batch_size, self.args.pred_len, self.text_embedding_dim)).to(self.device)
                text_dec_inp = torch.cat([prompt_emb[:, :self.args.label_len, :], text_dec_inp], dim=1).float().to(self.device) # 32 144 12
                dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1).detach()

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, loss_mmd = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, epoch)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # TODO: this only works for single variate time series
                outputs = outputs[:, :, 0].unsqueeze(-1)
                # outputs = (1-self.prompt_weight)*outputs+self.prompt_weight*prior_y
            

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y) + loss_mmd + F.l1_loss(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    model_optim_mlp.step()
                    model_optim_proj.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print(loss_mmd)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, _, _, _, _ = self.vali(test_data, test_loader, criterion, all_metric=True)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss (MSE): {4:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.mlp.eval()
        self.mlp_proj.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                prior_y=torch.from_numpy(test_data.get_prior_y(index)).float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_text=test_data.get_text(index)
                batch_text_flattened = batch_text_flattened = batch_text.reshape(-1).tolist()
                if self.Doc2Vec==False:
                    tokenized_output = self.tokenizer(
                        batch_text_flattened,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    language_max_len = tokenized_output['input_ids'].shape[1]
                    input_ids = tokenized_output['input_ids'].view(self.args.batch_size, self.args.seq_len, language_max_len).to(self.device)
                    attn_mask = tokenized_output['attention_mask'].view(self.args.batch_size, self.args.seq_len, language_max_len).to(self.device)
                    prompt_embeddings = self.llm_model.get_input_embeddings()(input_ids)

                else:
                    prompt = batch_text
                    prompt_embeddings = torch.tensor([self.text_model.infer_vector(text) for text in prompt]).to(self.device)
                if self.use_fullmodel:
                    prompt_emb =self.llm_model(inputs_embeds=prompt_embeddings).last_hidden_state
                else:
                    prompt_emb=prompt_embeddings

                if self.Doc2Vec == False:
                    # Expand attn_mask to match prompt_emb dimensions
                    expanded_mask = attn_mask.unsqueeze(-1).expand_as(prompt_emb)

                    if self.pool_type == "avg":
                        # Mask the embeddings by setting padded tokens to 0
                        masked_emb = prompt_emb * expanded_mask
                        valid_counts = expanded_mask.sum(dim=2, keepdim=True).clamp(min=1)
                        pooled_emb = masked_emb.sum(dim=2) / valid_counts.squeeze(2)
                        prompt_emb = pooled_emb

                    elif self.pool_type == "max":
                        # Mask the embeddings by setting padded tokens to a very small value
                        masked_emb = prompt_emb.masked_fill(expanded_mask == 0, float('-inf'))
                        pooled_emb, _ = masked_emb.max(dim=2)
                        prompt_emb = pooled_emb

                    elif self.pool_type == "min":
                        # Mask the embeddings by setting padded tokens to a very large value
                        masked_emb = prompt_emb.masked_fill(expanded_mask == 0, float('inf'))
                        pooled_emb, _ = masked_emb.min(dim=2)
                        prompt_emb = pooled_emb
                else:
                    prompt_emb = prompt_emb
                    
                prompt_emb = self.mlp(prompt_emb)  

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # batch_x is [bsz, seq_len, num_vars], prompt_emb is [bsz, seq_len, text_embedding_dim]. concatenate them in the last dimension
                batch_x = torch.cat([batch_x, prompt_emb], dim=-1).detach()

                # dec_inp is [bsz, label_len + pred_len, num_vars], where only label_len is the true data, the rest is 0
                text_dec_inp = torch.zeros((self.args.batch_size, self.args.pred_len, self.text_embedding_dim)).to(self.device)
                text_dec_inp = torch.cat([prompt_emb[:, :self.args.label_len, :], text_dec_inp], dim=1).float().to(self.device)
                dec_inp = torch.cat([dec_inp, text_dec_inp], dim=-1).detach()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,_ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, 1)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # TODO: this only works for single variate time series
                outputs = outputs[:, :, 0].unsqueeze(-1)
                # outputs = (1-self.prompt_weight)*outputs +self.prompt_weight*prior_y
                
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        
        # dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(self.args.save_name, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mse
