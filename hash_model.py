from timm.layers import SwiGLU
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import normalize
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class FusionTransFormer(nn.Module):
    def __init__(self, num_layers=1, hidden_size=1024, nhead=4, transformer_num_heads=8, transformer_ff_dim=2048):
        super(FusionTransFormer, self).__init__()
        self.d_model = hidden_size
        self.signal_d = int(self.d_model / 2)
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            batch_first=False
        )
        self.transformer = TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False
        )
        
        # Projection layers
        self.inproj = nn.Linear(self.signal_d, self.signal_d)
        self.outproj = nn.Linear(self.signal_d, self.signal_d)
        
        # Cross-attention layer for interaction
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.signal_d, 
            num_heads=nhead, 
            batch_first=False
        )
        
        # Normalization
        self.grn1 = GRN(dim=self.signal_d)
        self.grn2 = GRN(dim=self.d_model)

    def forward(self, img_cls, txt_eos):
        # Shortcut connections
        short_img_cls = self.inproj(img_cls)
        short_txt_eos = self.inproj(txt_eos)
        
        # Cross attention between modalities
        attn_output_img, _ = self.cross_attn(
            query=img_cls.unsqueeze(0),
            key=txt_eos.unsqueeze(0),
            value=txt_eos.unsqueeze(0))
        attn_output_txt, _ = self.cross_attn(
            query=txt_eos.unsqueeze(0),
            key=img_cls.unsqueeze(0),
            value=img_cls.unsqueeze(0))
        
        img_cls = self.outproj(self.grn1(attn_output_img).squeeze())
        txt_eos = self.outproj(self.grn1(attn_output_txt).squeeze())
        
        # Residual connections
        img_cls = 0.5 * img_cls + 0.5 * short_img_cls
        txt_eos = 0.5 * txt_eos + 0.5 * short_txt_eos
        
        # Combine and process through transformer
        res_temp_cls = torch.concat((img_cls, txt_eos), dim=-1)
        res_temp_cls = self.grn2(res_temp_cls)
        encoder_X = self.transformer(res_temp_cls.unsqueeze(0))
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=-1)
        
        img_cls, txt_eos = encoder_X_r[:, :self.signal_d], encoder_X_r[:, self.signal_d:]
        return img_cls, txt_eos



class MLPLayer(nn.Module):
    def __init__(self, dim_list, dropout=0., activation='relu'):
        super().__init__()

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        else:
            pass

        self.mlp = nn.Sequential()

        for i in range(len(dim_list) - 2):
            _in = dim_list[i]
            _out = dim_list[i + 1]
            
            self.mlp.add_module(f"linear_{i}", nn.Linear(_in, _out))
            if activation == 'swiglu':
                self.mlp.add_module(f"activate_{i}", SwiGLU(in_features=_out))
            else:    
                self.mlp.add_module(f"activate_{i}", self.activation_layer)
            self.mlp.add_module(f"dropout_{i}", nn.Dropout(p=dropout))

        self.mlp.add_module(f"linear_final", nn.Linear(dim_list[-2], dim_list[-1]))

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPs(nn.Module):

    def __init__(self, org_dim, hidden_dim, dropout=0., num_layers=2, activation='relu'):
        super().__init__()
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'gelu':
            self.activation_layer = nn.GELU()
        elif activation == 'swiglu':
            self.activation_layer = SwiGLU(in_features=hidden_dim)
        else:
            pass

        self.mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(org_dim, hidden_dim),
            self.activation_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, org_dim),
        ) for i in range(num_layers))

        self.lns = nn.ModuleList(nn.LayerNorm(org_dim) for i in range(num_layers))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x + self.mlps[i](self.lns[i](x))
        return x

class HashingEncoder(nn.Module):
    def __init__(self, org_dim, k_bits, ):
        super().__init__()
        self.fc = nn.Linear(org_dim, k_bits)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        return torch.tanh(self.fc(x))

class HashingDecoder(nn.Module):
    def __init__(self, org_bit_dim, recon_bit_dim):
        super().__init__()
        self.mlp = MLPLayer(dim_list=[org_bit_dim, recon_bit_dim, recon_bit_dim])
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        return torch.tanh(self.mlp(x))





class HashingModel2(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.dropout = dropout = args.dropout
        self.activation = activation = args.activation
        self.res_mlp_layers = res_mlp_layers = args.res_mlp_layers
        self.auxiliary_bit_dim = auxiliary_bit_dim = args.auxiliary_bit_dim
        self.transformer_layers = args.transformer_layers
        self.concept_num = args.concept_num

        clip_embed_dim = 512

        # Parse k_bits_list from string to list of integers
        self.k_bits_list = list(map(int, args.k_bits_list.split(",")))  # str -> list

        # Extend bits list with auxiliary bit dimension
        self.extend_bits_list = []
        self.extend_bits_list.extend(self.k_bits_list)
        self.extend_bits_list.append(self.auxiliary_bit_dim)

        # Shared residual MLPs for image, text, and prompt features
        self.resmlp_i = self.resmlp_t = self.resmlp_p = ResidualMLPs(
            org_dim=clip_embed_dim,
            hidden_dim=4 * clip_embed_dim,
            dropout=dropout,
            num_layers=res_mlp_layers,
            activation=activation
        )

        # Hashing encoders for different bit dimensions
        self.hash_encoders = nn.ModuleList(
            HashingEncoder(org_dim=clip_embed_dim, k_bits=one)
            for one in self.extend_bits_list
        )

        # Hashing decoders for reconstruction
        self.hash_decoders = nn.ModuleList(
            HashingDecoder(one, auxiliary_bit_dim)
            for one in self.k_bits_list
        )

        # Replace FusionTransMamba with FusionTransFormer
        self.FuseTransformer = FusionTransFormer(
            num_layers=1,
            hidden_size=clip_embed_dim * 2,
            nhead=4,
            transformer_num_heads=8,
            transformer_ff_dim=2048
        )

    def forward(self, img_cls, txt_eos, prompt_eos):
        output_dict = {}

        # Store short-term features for residual connection
        short_img_cls = img_cls
        short_txt_cls = txt_eos
        short_prompt_eos = prompt_eos

        # Process with FusionTransFormer
        img_cls, txt_eos = self.FuseTransformer(img_cls, txt_eos)
        img_cls, prompt_eos = self.FuseTransformer(img_cls, prompt_eos)

        # Residual connections
        img_cls = 0.5 * short_img_cls + 0.5 * img_cls
        txt_eos = 0.5 * short_txt_cls + 0.5 * txt_eos
        prompt_eos = 0.5 * short_prompt_eos + 0.5 * prompt_eos

        fuse_feature = img_cls + txt_eos 
        res_fuse_cls = self.resmlp_i(fuse_feature)
        res_prompt_cls = self.resmlp_p(prompt_eos)

        output_dict['res_fuse_cls'] = F.normalize(res_fuse_cls, dim=-1)
        output_dict['res_prompt_cls'] = F.normalize(res_prompt_cls, dim=-1)

        output_dict['fuse_cls_hash'] = {}
        output_dict['fuse_cls_hash_recon'] = {}
        
        # 生成哈希和重构
        for i, one in enumerate(self.extend_bits_list):
            fuse_cls_hash = self.hash_encoders[i](fuse_feature)
            output_dict['fuse_cls_hash'][one] = fuse_cls_hash
            
            if one != self.auxiliary_bit_dim:
                fuse_cls_hash_recon = self.hash_decoders[i](fuse_cls_hash)
                output_dict['fuse_cls_hash_recon'][one] = fuse_cls_hash_recon
        
        output_dict['res_img_cls'] = F.normalize(img_cls, dim=-1)
        output_dict['res_txt_cls'] = F.normalize(txt_eos, dim=-1)
        output_dict['res_prompt_cls'] = F.normalize(prompt_eos, dim=-1)


        return output_dict


# class EnhancedPromptLearner(nn.Module):
#     def __init__(self, clip_model, n_cls, maxWords, device, 
#                 use_image_context=True, use_text_context=True):
#         super().__init__()
#         self.device = device
#         self.maxWords = maxWords
        
#         # 获取 tokenizer 并添加特殊 token
#         self.tokenizer = open_clip.SimpleTokenizer()
        
#         # 添加特殊 token 到 tokenizer
#         special_tokens_to_add = {
#             "CLS_TOKEN": "<|startoftext|>", 
#             "SEP_TOKEN": "<|endoftext|>",
#             "IMG_TOKEN": "<image>",
#             "TXT_TOKEN": "<text>"
#         }
        
#         for token_name, token in special_tokens_to_add.items():
#             if token not in self.tokenizer.encoder:
#                 # 为新的特殊 token 分配一个 id
#                 new_id = max(self.tokenizer.encoder.values()) + 1
#                 self.tokenizer.encoder[token] = new_id
#                 self.tokenizer.decoder[new_id] = token
                
#         self.SPECIAL_TOKEN = special_tokens_to_add
        
#         clip_model.tokenizer = self.tokenizer
        
#         self.ctx_init = "This is an image containing"
#         self.use_image_context = use_image_context
#         self.use_text_context = use_text_context
        
#         self.templates = [
#             "An image showing", 
#             "A photo depicting",
#             "A picture of",
#             "This is an image with"
#         ]
        
#         with torch.no_grad():
#             self.token_embedding = clip_model.token_embedding.to(device)
#             for param in self.token_embedding.parameters():
#                 param.requires_grad = False
        
#         self.ctx_dim = clip_model.ln_final.weight.shape[0]
        
#         all_special_tokens = list(self.SPECIAL_TOKEN.values())
#         self.special_token_ids = torch.tensor(
#             [self.tokenizer.encoder[token] for token in all_special_tokens],
#             device=device, dtype=torch.long
#         )
        
        
#         # 初始化可学习的 context vectors
#         ctx_vectors = torch.empty(self.maxWords, self.ctx_dim, device=device)
#         nn.init.normal_(ctx_vectors, std=0.02)
#         self.ctx = nn.Parameter(ctx_vectors)
        
#         # 预分配 buffer
#         self.register_buffer('padding_zeros', torch.zeros(self.maxWords, dtype=torch.long, device=device))
#         self.batch_buffer = None
        
#         # 改进的 prompt 缓存 (LRU 缓存)
#         self._prompt_cache = {}
#         self.cache_capacity = 1000
    
#     def generate_dynamic_prompt(self, name_list, image_features=None, text_features=None):
#         """
#         生成动态 prompt，结合图像/文本上下文信息
#         """
#         # 随机选择模板增加多样性
#         template = np.random.choice(self.templates)
        
#         # 处理类别名
#         if len(name_list) > 1:
#             names = ", ".join(n.replace("_", " ") for n in name_list[:-1])
#             names += f" and {name_list[-1].replace('_', ' ')}"
#         else:
#             names = name_list[0].replace("_", " ")
        
#         # 基础 prompt
#         prompt_parts = [f"{template} {names}"]
        
#         # 添加图像上下文信息
#         if self.use_image_context and image_features is not None:
#             prompt_parts.append(f"{self.SPECIAL_TOKEN['IMG_TOKEN']} shows visual details")
        
#         # 添加文本上下文信息
#         if self.use_text_context and text_features is not None:
#             prompt_parts.append(f"{self.SPECIAL_TOKEN['TXT_TOKEN']} provides textual context")
        
#         return " ".join(prompt_parts)
    
#     def build_prompt_ids(self, prompt_text):
#         """
#         将 prompt 文本转换为 token ids
#         """
#         tokens = self.tokenizer.tokenize(prompt_text)
#         token_ids = torch.tensor(
#             self.tokenizer.convert_tokens_to_ids(tokens[:self.maxWords-2]),
#             device=self.device, dtype=torch.long
#         )
        
#         # 使用预分配 buffer
#         prompt_ids = self.padding_zeros.clone()
#         prompt_ids[0] = self.special_token_ids[0]  # CLS
#         length = token_ids.size(0)
#         prompt_ids[1:length+1].copy_(token_ids)
#         prompt_ids[length+1] = self.special_token_ids[1]  # SEP
        
#         return prompt_ids
    
#     @torch.cuda.amp.autocast()
#     def forward(self, classnames, image_features=None, text_features=None):
#         batch_size = len(classnames)
        
#         # 调整 batch buffer 大小
#         if self.batch_buffer is None or self.batch_buffer.size(0) != batch_size:
#             self.batch_buffer = torch.empty(
#                 (batch_size, self.maxWords),
#                 dtype=torch.long,
#                 device=self.device
#             )
        
#         # 批量生成动态 prompt
#         for i, name_list in enumerate(classnames):
#             cache_key = tuple(name_list)
            
#             # 修改为更安全的特征处理方式
#             feat_key = []
#             if image_features is not None:
#                 # 使用特征的统计信息作为key，确保转换为可迭代的tuple
#                 mean_val = image_features[i].mean().item()
#                 std_val = image_features[i].std().item()
#                 feat_key.append((mean_val, std_val))
            
#             if text_features is not None:
#                 mean_val = text_features[i].mean().item()
#                 std_val = text_features[i].std().item()
#                 feat_key.append((mean_val, std_val))
            
#             cache_key += tuple(feat_key)
            
#             if cache_key in self._prompt_cache:
#                 self.batch_buffer[i] = self._prompt_cache[cache_key]
#             else:
#                 # 修改为更安全的特征传递方式
#                 img_feat = image_features[i] if image_features is not None else None
#                 txt_feat = text_features[i] if text_features is not None else None
                
#                 prompt_text = self.generate_dynamic_prompt(name_list, img_feat, txt_feat)
#                 prompt_ids = self.build_prompt_ids(prompt_text)
#                 self.batch_buffer[i] = prompt_ids
                
#                 # 更新缓存 (LRU 策略)
#                 if len(self._prompt_cache) >= self.cache_capacity:
#                     self._prompt_cache.pop(next(iter(self._prompt_cache)))
#                 self._prompt_cache[cache_key] = prompt_ids
        
#         prompts = self.batch_buffer[:batch_size]
        
#         # 计算 embedding
#         embedding = self.token_embedding(prompts)
#         ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
        
#         return embedding + ctx
    

# class PromptHash(nn.Module):
#     def __init__(self, class_name_list, layers_to_unfreeze, args=None):
#         super(PromptHash, self).__init__()
#         self.args = args
        
#         # 根据数据集设置类别数
#         self.n_cls = {
#             "coco": 80,
#             "flickr25k": 24,
#             "nuswide": 21
#         }.get(args.dataset, 80)
        
#         # 初始化CLIP模型
#         self.clip, _, _ = open_clip.create_model_and_transforms(
#             'ViT-B-16-quickgelu', 
#             pretrained='metaclip_fullcc'
#         )
#         self.class_name_list = class_name_list

#         # 冻结CLIP参数
#         if self.args.is_freeze_clip:
#             for param in self.clip.parameters():
#                 param.requires_grad = False
        
#         # 解冻指定层
#         for name, param in self.clip.named_parameters():
#             if name in layers_to_unfreeze:
#                 param.requires_grad = True
#                 print(f"Unfrozen layer: {name}")

#         # 使用改进的Prompt Learner
#         self.prompt_learner = EnhancedPromptLearner(
#             clip_model=self.clip,
#             n_cls=self.n_cls,
#             maxWords=args.max_words,
#             device=args.rank,
#             use_image_context=True,
#             use_text_context=True
#         )

#         # 哈希模型保持不变
#         self.hash = HashingModel2(args=args)
        
#         self.feature_extractor = nn.Sequential(
#                     nn.Linear(512, 512),
#                     nn.ReLU(),
#                     nn.LayerNorm(512)
#                 )
        
#     # def extract_context_features(self, image, text):
#     #     """提取图像和文本的上下文特征"""
#     #     with torch.no_grad():
#     #         # 图像特征提取
#     #         image_prompt = torch.randn(1, 1, 1, 1, device=image.device)
#     #         img_feat = self.clip.encode_image(image, image_prompt)
            
#     #         # 文本特征提取 - 处理可能的 tuple 输出
#     #         dummy_text_prompt = torch.zeros(
#     #             text.size(0), 
#     #             self.args.max_words, 
#     #             self.prompt_learner.ctx_dim,
#     #             device=image.device
#     #         )
#     #         txt_output = self.clip.encode_text(text, dummy_text_prompt)
            
#     #         # 如果返回的是 tuple，取第一个元素 (通常是特征)
#     #         if isinstance(txt_output, tuple):
#     #             txt_feat = txt_output[0]
#     #         else:
#     #             txt_feat = txt_output
        
#     #     # 特征处理
#     #     img_context = self.feature_extractor(img_feat)
#     #     txt_context = self.feature_extractor(txt_feat)
#     #     return img_context, txt_context
#     def extract_context_features(self, image, text):
#         """提取图像和文本的上下文特征"""
#         with torch.no_grad():
#             # 图像特征提取
#             image_prompt = torch.randn(1, 1, 1, 1, device=image.device)
#             img_feat = self.clip.encode_image(image, image_prompt)
#             # 文本特征提取
#             dummy_text_prompt = torch.zeros(
#                 text.size(0), 
#                 self.args.max_words, 
#                 self.prompt_learner.ctx_dim,
#                 device=image.device
#             )
#             txt_output = self.clip.encode_text(text, dummy_text_prompt)
#             # 如果返回的是 tuple，取第一个元素 (通常是特征)
#             if isinstance(txt_output, tuple):
#                 txt_feat = txt_output[0]
#             else:
#                 txt_feat = txt_output
#         # 特征处理
#         img_context = self.feature_extractor(img_feat)
#         txt_context = self.feature_extractor(txt_feat)
#         return img_context, txt_context
    


#     def forward(self, image, text, label):
#         B = image.size(0)
        
#         # 1. 获取类别对应的prompt
#         indices = [torch.where(label[i] == 1)[0] for i in range(B)]
#         class_names_prompt = [[self.class_name_list[j] for j in indices[i]] for i in range(B)]
        
#         # 2. 提取上下文特征
#         img_context, txt_context = self.extract_context_features(image, text)
        
#         # 3. 生成增强prompt
#         text_prompt = self.prompt_learner(
#             class_names_prompt,
#             image_features=img_context,
#             text_features=txt_context
#         )
        
#         # 4. 编码图像和文本
#         image_prompt = torch.randn(1, 1, 1, 1, device=image.device)
#         img_cls = self.clip.encode_image(image, image_prompt)
        
#         # 处理文本编码输出
#         text_output = self.clip.encode_text(text, text_prompt)
#         if isinstance(text_output, tuple):
#             txt_eos, prompt_eos = text_output[0], text_output[1]
#         else:
#             txt_eos = text_output
#             prompt_eos = None
        
#         # 5. 通过哈希模型
#         output_dict = self.hash(img_cls, txt_eos, prompt_eos if prompt_eos is not None else txt_eos)
        
#         return output_dict



class StaticPromptLearner(nn.Module):
    def __init__(self, clip_model, n_cls, maxWords, device):
        super().__init__()
        self.device = device
        self.maxWords = maxWords
        
        # 保留基础 tokenizer 配置
        self.tokenizer = open_clip.SimpleTokenizer()
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>", 
            "SEP_TOKEN": "<|endoftext|>"
        }
        
        # 固定模板（静态 prompt 核心）
        self.templates = [
            "An image showing {}", 
            "A photo depicting {}",
            "A picture of {}",
            "This is an image with {}"
        ]
        
        # 初始化 CLIP 相关参数
        with torch.no_grad():
            self.token_embedding = clip_model.token_embedding.to(device)
            for param in self.token_embedding.parameters():
                param.requires_grad = False
        
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.special_token_ids = torch.tensor(
            [self.tokenizer.encoder[self.SPECIAL_TOKEN["CLS_TOKEN"]],
             self.tokenizer.encoder[self.SPECIAL_TOKEN["SEP_TOKEN"]]],
            device=device, dtype=torch.long
        )
        
        # 可学习的 context vectors（可选，若需保留少量可学习参数）
        ctx_vectors = torch.empty(self.maxWords, self.ctx_dim, device=device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # 缓存和缓冲区
        self.register_buffer('padding_zeros', torch.zeros(self.maxWords, dtype=torch.long, device=device))
        self.batch_buffer = None
        self._prompt_cache = {}  # 仅基于类别名称缓存
        self.cache_capacity = 1000
    
    def generate_static_prompt(self, name_list):
        """仅基于类别名称生成固定模板的 prompt"""
        # 处理类别名称
        if len(name_list) > 1:
            names = ", ".join(n.replace("_", " ") for n in name_list[:-1])
            names += f" and {name_list[-1].replace('_', ' ')}"
        else:
            names = name_list[0].replace("_", " ")
        
        # 随机选择固定模板（不依赖图像/文本特征）
        template = np.random.choice(self.templates)
        return f"{template} {names}"
    
    def build_prompt_ids(self, prompt_text):
        """与原逻辑一致，将文本转换为 token ids"""
        tokens = self.tokenizer.tokenize(prompt_text)
        token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(tokens[:self.maxWords-2]),
            device=self.device, dtype=torch.long
        )
        
        prompt_ids = self.padding_zeros.clone()
        prompt_ids[0] = self.special_token_ids[0]  # CLS
        length = token_ids.size(0)
        prompt_ids[1:length+1].copy_(token_ids)
        prompt_ids[length+1] = self.special_token_ids[1]  # SEP
        return prompt_ids
    
    def forward(self, classnames):
        """仅接收类别名称，不依赖图像/文本特征"""
        batch_size = len(classnames)
        
        if self.batch_buffer is None or self.batch_buffer.size(0) != batch_size:
            self.batch_buffer = torch.empty(
                (batch_size, self.maxWords),
                dtype=torch.long,
                device=self.device
            )
        
        # 仅基于类别名称生成 prompt（无特征依赖）
        for i, name_list in enumerate(classnames):
            cache_key = tuple(name_list)  # 缓存 key 仅为类别名称
            
            if cache_key in self._prompt_cache:
                self.batch_buffer[i] = self._prompt_cache[cache_key]
            else:
                prompt_text = self.generate_static_prompt(name_list)
                prompt_ids = self.build_prompt_ids(prompt_text)
                self.batch_buffer[i] = prompt_ids
                
                # 更新缓存
                if len(self._prompt_cache) >= self.cache_capacity:
                    self._prompt_cache.pop(next(iter(self._prompt_cache)))
                self._prompt_cache[cache_key] = prompt_ids
        
        prompts = self.batch_buffer[:batch_size]
        embedding = self.token_embedding(prompts)
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)
        return embedding + ctx

class PromptHash(nn.Module):  # 新类名，区分原动态版本
    def __init__(self, class_name_list, layers_to_unfreeze, args=None):
        super(PromptHash, self).__init__()
        self.args = args
        
        self.n_cls = {
            "coco": 80,
            "flickr25k": 24,
            "nuswide": 21
        }.get(args.dataset, 80)
        
        # 初始化 CLIP 模型（与原逻辑一致）
        self.clip, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16-quickgelu', 
            pretrained='metaclip_fullcc'
        )
        self.class_name_list = class_name_list

        # 冻结/解冻 CLIP 参数（与原逻辑一致）
        if self.args.is_freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        for name, param in self.clip.named_parameters():
            if name in layers_to_unfreeze:
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")

        # 使用静态 prompt 生成器
        self.prompt_learner = StaticPromptLearner(  # 替换为静态版本
            clip_model=self.clip,
            n_cls=self.n_cls,
            maxWords=args.max_words,
            device=args.rank
        )

        # 哈希模型保持不变
        self.hash = HashingModel2(args=args)
        
        self.feature_extractor = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.LayerNorm(512)
                )
    
    def forward(self, image, text, label):
        B = image.size(0)
        
        # 1. 获取类别对应的 prompt（仅基于类别名称，无特征依赖）
        indices = [torch.where(label[i] == 1)[0] for i in range(B)]
        class_names_prompt = [[self.class_name_list[j] for j in indices[i]] for i in range(B)]
        
        # 2. 生成静态 prompt（无需上下文特征）
        text_prompt = self.prompt_learner(class_names_prompt)  # 仅传入类别名称
        
        # 3. 编码图像和文本（与原逻辑一致）
        image_prompt = torch.randn(1, 1, 1, 1, device=image.device)
        img_cls = self.clip.encode_image(image, image_prompt)
        
        text_output = self.clip.encode_text(text, text_prompt)
        if isinstance(text_output, tuple):
            txt_eos, prompt_eos = text_output[0], text_output[1]
        else:
            txt_eos = text_output
            prompt_eos = txt_eos  # 若仅返回单值，用文本特征替代
        
        # 4. 通过哈希模型
        output_dict = self.hash(img_cls, txt_eos, prompt_eos)
        
        return output_dict

