from hash_model import PromptHash
import time
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn.functional as F
from optimization import BertAdam
from load_data import generate_dataset
from utilss import * 

dataset_root_path = "./dataset"

class TrainerAsym:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset

        class_name_map = {
            "coco": coco_class_name_list,
            "flickr25k": flickr25k_class_name_list,
            "nuswide": nuswide_class_name_list,
        }
        self.class_name_list = class_name_map.get(self.dataset_name, [])
        if not self.class_name_list:
            print(f"Warning: Class name list for dataset '{self.dataset_name}' not found.")


        self.accumulation_steps = args.accumulation_steps

        self.early_stop_patience = getattr(args, 'early_stop_patience', 10) # Use args if available, else default
        self.early_stop_counter = 0

        set_seed(self.args.seed)

        os.makedirs(self.args.save_dir, exist_ok=True)
        log_file_name = "train.log" if self.args.is_train else "test.log"
        self.logger = get_logger(os.path.join(self.args.save_dir, log_file_name))

        self.logger.info('Start logging...')

        if self.args.is_train:
            log_str_args = "\n" + "="*20 + " Arguments " + "="*20 + "\n"
            for k, v in sorted(vars(self.args).items()): # Iterate over sorted args for consistent logging
                log_str_args += f"{k:<40} = {v}\n"
            log_str_args += "="*53 + "\n"
            self.logger.info(log_str_args)
        else:
            self.logger.info(f"Pretrained model path: {self.args.pretrained}")

        self.rank = self.args.rank  # gpu rank
        self.device = torch.device("cuda", self.rank)

        self._init_dataset()
        self._init_model()

        self.best_avg_map = 0.0
        self.best_epoch = 0

        self.logger.info(f"Train dataset length: {len(self.train_loader.dataset)}")

        self.k_bits_list = list(map(int, self.args.k_bits_list.split(",")))

        self.fbuf = {}
        self.bbuf = {}

        self.extend_bits_list = self.k_bits_list + [self.args.auxiliary_bit_dim]

        for one_bit_length in self.extend_bits_list:
            self.fbuf[one_bit_length] = torch.randn(self.args.train_num, one_bit_length).to(self.device, non_blocking=True)
            self.bbuf[one_bit_length] = torch.sign(self.fbuf[one_bit_length])

        if self.args.is_train:
            self.train()

    def _init_model(self):
        self.logger.info("Initializing model...")
        self.logger.info("Using ViT & GPT2 based PromptHash.")

        # Layers to unfreeze (kept as is from original, ensure this list is correct for your model)
        self.layers_to_unfreeze = [
            # Visual encoder last two blocks
            'visual.transformer.resblocks.10.ln_1.weight', 'visual.transformer.resblocks.10.ln_1.bias',
            'visual.transformer.resblocks.10.attn.in_proj_weight', 'visual.transformer.resblocks.10.attn.in_proj_bias',
            'visual.transformer.resblocks.10.attn.out_proj.weight', 'visual.transformer.resblocks.10.attn.out_proj.bias',
            'visual.transformer.resblocks.10.ln_2.weight', 'visual.transformer.resblocks.10.ln_2.bias',
            'visual.transformer.resblocks.10.mlp.c_fc.weight', 'visual.transformer.resblocks.10.mlp.c_fc.bias',
            'visual.transformer.resblocks.10.mlp.c_proj.weight', 'visual.transformer.resblocks.10.mlp.c_proj.bias',
            'visual.transformer.resblocks.11.ln_1.weight', 'visual.transformer.resblocks.11.ln_1.bias',
            'visual.transformer.resblocks.11.attn.in_proj_weight', 'visual.transformer.resblocks.11.attn.in_proj_bias',
            'visual.transformer.resblocks.11.attn.out_proj.weight', 'visual.transformer.resblocks.11.attn.out_proj.bias',
            'visual.transformer.resblocks.11.ln_2.weight', 'visual.transformer.resblocks.11.ln_2.bias',
            'visual.transformer.resblocks.11.mlp.c_fc.weight', 'visual.transformer.resblocks.11.mlp.c_fc.bias',
            'visual.transformer.resblocks.11.mlp.c_proj.weight', 'visual.transformer.resblocks.11.mlp.c_proj.bias',
            # Text encoder last two blocks
            'transformer.resblocks.10.ln_1.weight', 'transformer.resblocks.10.ln_1.bias',
            'transformer.resblocks.10.attn.in_proj_weight', 'transformer.resblocks.10.attn.in_proj_bias',
            'transformer.resblocks.10.attn.out_proj.weight', 'transformer.resblocks.10.attn.out_proj.bias',
            'transformer.resblocks.10.ln_2.weight', 'transformer.resblocks.10.ln_2.bias',
            'transformer.resblocks.10.mlp.c_fc.weight', 'transformer.resblocks.10.mlp.c_fc.bias',
            'transformer.resblocks.10.mlp.c_proj.weight', 'transformer.resblocks.10.mlp.c_proj.bias',
            'transformer.resblocks.11.ln_1.weight', 'transformer.resblocks.11.ln_1.bias',
            'transformer.resblocks.11.attn.in_proj_weight', 'transformer.resblocks.11.attn.in_proj_bias',
            'transformer.resblocks.11.attn.out_proj.weight', 'transformer.resblocks.11.attn.out_proj.bias',
            'transformer.resblocks.11.ln_2.weight', 'transformer.resblocks.11.ln_2.bias',
            'transformer.resblocks.11.mlp.c_fc.weight', 'transformer.resblocks.11.mlp.c_fc.bias',
            'transformer.resblocks.11.mlp.c_proj.weight', 'transformer.resblocks.11.mlp.c_proj.bias',
            # Text prompt encoder
            'prompt_transformer.resblocks.0.ln_1.weight', 'prompt_transformer.resblocks.0.ln_1.bias',
            'prompt_transformer.resblocks.0.attn.in_proj_weight', 'prompt_transformer.resblocks.0.attn.in_proj_bias',
            'prompt_transformer.resblocks.0.attn.out_proj.weight', 'prompt_transformer.resblocks.0.attn.out_proj.bias',
            'prompt_transformer.resblocks.0.ln_2.weight', 'prompt_transformer.resblocks.0.ln_2.bias',
            'prompt_transformer.resblocks.0.mlp.c_fc.weight', 'prompt_transformer.resblocks.0.mlp.c_fc.bias',
            'prompt_transformer.resblocks.0.mlp.c_proj.weight', 'prompt_transformer.resblocks.0.mlp.c_proj.bias',
            'prompt_transformer.resblocks.1.ln_1.weight', 'prompt_transformer.resblocks.1.ln_1.bias',
            'prompt_transformer.resblocks.1.attn.in_proj_weight', 'prompt_transformer.resblocks.1.attn.in_proj_bias',
            'prompt_transformer.resblocks.1.attn.out_proj.weight', 'prompt_transformer.resblocks.1.attn.out_proj.bias',
            'prompt_transformer.resblocks.1.ln_2.weight', 'prompt_transformer.resblocks.1.ln_2.bias',
            'prompt_transformer.resblocks.1.mlp.c_fc.weight', 'prompt_transformer.resblocks.1.mlp.c_fc.bias',
            'prompt_transformer.resblocks.1.mlp.c_proj.weight', 'prompt_transformer.resblocks.1.mlp.c_proj.bias'
        ]
        self.model = PromptHash(class_name_list=self.class_name_list, layers_to_unfreeze=self.layers_to_unfreeze, args=self.args).to(self.device)

        if self.args.pretrained and os.path.exists(self.args.pretrained):
            self.logger.info(f"Loading pretrained model from {self.args.pretrained}")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=self.device))
        
        self.model.float() # Ensure model is in float32

        # Optimizer parameter groups
        clip_params_frozen = []
        clip_params_unfrozen = []
        for name, param in self.model.clip.named_parameters():
            if name in self.layers_to_unfreeze:
                clip_params_unfrozen.append(param)
            else:
                param.requires_grad_(False) # Explicitly freeze layers not in unfreeze list
                clip_params_frozen.append(param)


        fuse_transformer_params = list(self.model.hash.FuseTransformer.parameters())
        hash_other_params = [
            p for n, p in self.model.hash.named_parameters()
            if 'FuseTransformer' not in n and p.requires_grad
        ]
        
        param_groups = [
            {'params': clip_params_unfrozen, 'lr': self.args.clip_lr * 10, 'weight_decay': self.args.weight_decay},
            {'params': fuse_transformer_params, 'lr': self.args.lr * 0.1, 'weight_decay': self.args.weight_decay},
            {'params': hash_other_params, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        ]
        
        num_optimizer_steps_per_epoch = len(self.train_loader) // self.args.accumulation_steps
        total_optimizer_steps = num_optimizer_steps_per_epoch * self.args.epochs


        self.optimizer = BertAdam(
            param_groups,
            lr=self.args.lr, # Base LR, specific group LRs will override this
            warmup=self.args.warmup_proportion,
            schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6,
            t_total=total_optimizer_steps, # Corrected t_total
            weight_decay=self.args.weight_decay, # Default weight decay, overridden by group if specified
            max_grad_norm=1.0
        )

    def _init_dataset(self):
        self.logger.info("Initializing dataset...")
        self.logger.info(f"Using {self.args.dataset} dataset.")

        # Construct full paths for dataset files
        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)

        train_data, query_data, retrieval_data = generate_dataset(
            captionFile=self.args.caption_file,
            indexFile=self.args.index_file,
            labelFile=self.args.label_file,
            dataset_name=self.dataset_name,
            maxWords=self.args.max_words,
            imageResolution=self.args.resolution,
            query_num=self.args.query_num,
            train_num=self.args.train_num,
            seed=self.args.seed
        )

        self.train_labels = train_data.get_all_label().float()
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()

        self.args.retrieval_num = len(self.retrieval_labels)
        self.args.num_class = self.query_labels.shape[1]
        self.args.train_num = len(self.train_labels) # Update train_num based on actual loaded data

        self.logger.info(f"Query samples: {self.query_labels.shape[0]}, Retrieval samples: {self.retrieval_labels.shape[0]}")
        self.logger.info(f"Number of classes: {self.args.num_class}")

        self.train_loader = self._create_dataloader(train_data, shuffle=True)
        self.query_loader = self._create_dataloader(query_data, shuffle=False)
        self.retrieval_loader = self._create_dataloader(retrieval_data, shuffle=False)

    def _create_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            prefetch_factor=2 if self.args.num_workers > 0 else None # prefetch_factor requires num_workers > 0
        )

    def _change_model_state(self, mode="train"):
        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()

    def train(self):
        self.logger.info("Starting training...")

        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            self._change_model_state(mode="train")
            self.logger.info(f"\n\n{'='*30} Train Epoch: {epoch+1}/{self.args.epochs} {'='*30}")
            
            epoch_loss_dict = {'all_loss': 0.0} # Initialize with float

            for i, (image, text, key_padding_mask, label, index) in enumerate(self.train_loader):
                image = image.float().to(self.device, non_blocking=True)
                text = text.to(self.device, non_blocking=True)
                label = label.float().to(self.device, non_blocking=True)

                output_dict = self.model(image, text, label) # Assuming key_padding_mask is handled internally or not needed

                _B_batch = {}
                for bit_len in self.extend_bits_list:
                    fuse_hash = output_dict['fuse_cls_hash'][bit_len]
                    self.fbuf[bit_len][index] = fuse_hash.detach()
                    _B_batch[bit_len] = self.bbuf[bit_len][index] # Using bbuf which is sign of fbuf from previous epoch or init

                current_batch_losses = self.compute_loss(output_dict, label, _B_batch)
                loss = sum(current_batch_losses.values())

                for key, value in current_batch_losses.items():
                    epoch_loss_dict[key] = epoch_loss_dict.get(key, 0.0) + value.item()
                epoch_loss_dict['all_loss'] += loss.item()

                loss = loss / self.accumulation_steps # Normalize loss for accumulation
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            for bit_len in self.extend_bits_list:
                self.bbuf[bit_len] = torch.sign(self.fbuf[bit_len])

            avg_epoch_loss = epoch_loss_dict['all_loss'] / len(self.train_loader)
            self.logger.info(f">>>>>> [{epoch+1}/{self.args.epochs}] Avg All Loss: {avg_epoch_loss:.6f}")
            self.logger.info(f"LR: {'-'.join([f'{lr:.9f}' for lr in sorted(list(set(self.optimizer.get_lr())))])}")
            
            epoch_duration = int(time.time() - epoch_start_time)
            self.logger.info(f"Train Epoch [{epoch+1}] duration: {epoch_duration // 60} min, {epoch_duration % 60} sec")

            if (epoch + 1) % self.args.valid_freq == 0:
                val_start_time = time.time()
                current_avg_map, q_codes_dict, r_codes_dict = self.valid(epoch)
                
                if current_avg_map > self.best_avg_map:
                    self.best_avg_map = current_avg_map
                    self.best_epoch = epoch + 1
                    self.early_stop_counter = 0
                    self.logger.info(f"$$$$ New Best Avg mAP: {self.best_avg_map:.5f} at Epoch {self.best_epoch} $$$$")
                    # self.save_model(epoch + 1)
                    # for bit_len in self.extend_bits_list: # Save .mat for all, including auxiliary
                    #     file_name = f"Ours_{bit_len}-{self.args.dataset}_bits.mat"
                    #     self.save_mat(q_codes_dict[bit_len], r_codes_dict[bit_len], file_name)
                else:
                    self.early_stop_counter += 1
                    self.logger.info(f"No improvement in Avg mAP. EarlyStop Counter: {self.early_stop_counter}/{self.early_stop_patience}")

                val_duration = int(time.time() - val_start_time)
                self.logger.info(f"Validation after Epoch [{epoch+1}] duration: {val_duration // 60} min, {val_duration % 60} sec")

                if self.early_stop_counter >= self.early_stop_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                    break
        
        self.logger.info(f">>>>>>> Training Finished for {self.args.dataset}_{self.args.k_bits_list}. <<<<<<<")
        self.logger.info(f"Best Epoch: {self.best_epoch}, Best Avg mAP: {self.best_avg_map:.5f}")

    def valid(self, epoch): # epoch is passed for logging purposes
        self.logger.info(f"\n{'='*30} Validation after Epoch: {epoch+1}/{self.args.epochs} {'='*30}")
        self._change_model_state(mode="valid")

        with torch.no_grad():
            q_codes_dict = self.get_code(self.query_loader, self.args.query_num)
            r_codes_dict = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        sum_map_for_average = 0.0
        num_map_components = 0

        for bit_len in self.extend_bits_list:
            q_code = q_codes_dict[bit_len]
            r_code = r_codes_dict[bit_len]
            
            map_value = calc_map_k(
                q_code, r_code,
                self.query_labels.to(self.device, non_blocking=True),
                self.retrieval_labels.to(self.device, non_blocking=True)
            ).item()
            
            self.logger.info(f">>>>>> [{epoch+1}/{self.args.epochs}] {bit_len} bits - mAP: {map_value:.5f}")
            
            if bit_len != self.args.auxiliary_bit_dim: # Only average mAP for main k-bit lists
                sum_map_for_average += map_value
                num_map_components += 1
        
        current_epoch_avg_map = 0.0
        if num_map_components > 0:
             current_epoch_avg_map = sum_map_for_average / num_map_components # Standard average
             self.logger.info(f"Avg mAP over {self.k_bits_list} bits (standard avg): {current_epoch_avg_map:.5f}")
        else:
            self.logger.warning("No mAP components to average. k_bits_list might be empty or only contain auxiliary_bit_dim.")


        self.logger.info(f"Current Best Avg mAP: {self.best_avg_map:.5f} (from Epoch {self.best_epoch})")
        return current_epoch_avg_map, q_codes_dict, r_codes_dict


    def hash_loss_group(self, h_output, h_buffer_epoch_start, label_similarity, weight=1.0, bit_type='unknown_bits'):
        loss = weight * self.args.hyper_cls_sum * self.bayesian_loss(h_buffer_epoch_start, h_output, label_similarity)
        return {f'bayesian_{bit_type}': loss}


    def compute_loss(self, output_dict, current_batch_label, B_batch_epoch_start):
        all_losses = {}

        label_similarity = calc_neighbor(self.train_labels.to(self.device, non_blocking=True), current_batch_label)

        weights_map = {bit_len: 1.0 for bit_len in self.k_bits_list}
        weights_map[self.args.auxiliary_bit_dim] = self.args.mu

        for bit_len in self.extend_bits_list:
            fuse_cls_hash_output = output_dict['fuse_cls_hash'][bit_len]
            loss_group = self.hash_loss_group(
                fuse_cls_hash_output,
                self.fbuf[bit_len], 
                label_similarity,
                weight=weights_map[bit_len],
                bit_type=f'{bit_len}bits'
            )
            all_losses.update(loss_group)

        # Alignment Losses
        res_img_cls = output_dict['res_img_cls']
        res_txt_eos = output_dict['res_txt_cls'] 
        res_fuse_cls = output_dict['res_fuse_cls']
        res_prompt_eos = output_dict['res_prompt_cls']

        all_losses['Local_Prompt_Alignment'] = self.args.hyper_local * self.prompt_alignment_loss(
            res_fuse_cls, res_prompt_eos, temperature=self.args.tao_local
        )
        all_losses['Global_Prompt_Alignment'] = self.args.hyper_global * self.prompt_alignment_loss(
            res_img_cls, res_txt_eos, temperature=self.args.tao_global
        )

        target_recon_codes = B_batch_epoch_start[self.args.auxiliary_bit_dim]
        for bit_len in self.k_bits_list:
            fuse_cls_hash_recon_output = output_dict['fuse_cls_hash_recon'][bit_len]
            recon_loss = F.mse_loss(fuse_cls_hash_recon_output, target_recon_codes, reduction='sum') / fuse_cls_hash_recon_output.shape[0]
            all_losses[f'recon_f_{bit_len}'] = self.args.mu * self.args.hyper_recon * recon_loss
            
        return all_losses

    @torch.no_grad() # Ensure no gradients are computed during code generation
    def get_code(self, data_loader, num_samples: int):
        self._change_model_state(mode="valid") # Ensure model is in eval mode

        generated_codes_buffer = {
            bit_len: torch.empty(num_samples, bit_len, dtype=torch.float, device=self.device)
            for bit_len in self.extend_bits_list
        }

        for image, text, key_padding_mask, label, index in tqdm(data_loader, desc="Generating Codes"):
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            output_dict = self.model(image, text, label) # Pass label if model's forward pass expects it even for inference

            for bit_len in self.extend_bits_list:
                fuse_hash_output = output_dict['fuse_cls_hash'][bit_len] # .detach() is implicitly handled by torch.no_grad()
                generated_codes_buffer[bit_len][index.cpu().numpy()] = torch.sign(fuse_hash_output)
        
        return generated_codes_buffer

    def save_model(self, epoch_num):
        save_path = os.path.join(self.args.save_dir, f"model_epoch.pth")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Model saved to {save_path}")
        if epoch_num == self.best_epoch:
             torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model_best.pth"))
             self.logger.info(f"Best model (epoch {self.best_epoch}) also saved as model_best.pth")


    def save_mat(self, query_codes, retrieval_codes, file_name):
        save_dir = os.path.join(self.args.save_dir, "PR_curve_data")
        os.makedirs(save_dir, exist_ok=True)
        
        result_dict = {
            'val_B': query_codes.cpu().numpy(),
            'retrieval_B': retrieval_codes.cpu().numpy(),
            'L_te': self.query_labels.cpu().numpy(),    
            'L_db': self.retrieval_labels.cpu().numpy() 
        }
        
        full_save_path = os.path.join(save_dir, file_name)
        scio.savemat(full_save_path, result_dict)
        self.logger.info(f".mat file saved to {full_save_path}")
    
    def prompt_alignment_loss(self, out_1, out_2, temperature=0.07):
        out_1 = F.normalize(out_1, p=2, dim=1)
        out_2 = F.normalize(out_2, p=2, dim=1)
        
        batch_size = out_1.size(0)
        targets = torch.arange(batch_size, device=out_1.device)

        scores = out_1 @ out_2.t() / temperature
        
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores.t(), targets)

        return 0.5 * (loss0 + loss1)

    def bayesian_loss(self, features_a: torch.Tensor, features_b: torch.Tensor, label_similarity: torch.Tensor):
        s = 0.5 * torch.matmul(features_b, features_a.t()).clamp(min=-64, max=64) # (B,D) @ (D,N) -> (B,N)
        loss = -torch.mean(label_similarity * s - torch.log(1 + torch.exp(s)))
        return loss

if __name__ == "__main__":

    args = get_args() # You need to define/import get_args()
    trainer = TrainerAsym(args)