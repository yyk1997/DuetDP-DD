import os
import logging
import torch
import copy
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import pickle
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import tqdm
from tqdm import tqdm
from itertools import cycle
from PIL import Image


from models.DP_Diffusion.model.ncsnpp import NCSNpp
from models.DP_Diffusion.utils.util import set_seeds, make_dir, save_checkpoint, sample_random_image_batch, compute_fid
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from models.DP_Diffusion.model.ema import ExponentialMovingAverage
from models.DP_Diffusion.score_losses import EDMLoss, VPSDELoss, VESDELoss, VLoss
from models.DP_Diffusion.denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, VDenoiser
from models.DP_Diffusion.samplers import ddim_sampler, edm_sampler
from models.DP_Diffusion.generate_base import generate_batch

import importlib
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader, TensorDataset

opacus = importlib.import_module('opacus')

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from models.synthesizer import DPSynther

def cosine_similarity(ta, tb):
    bs1, bs2 = ta.shape[0], tb.shape[0]
    frac_up = torch.matmul(ta, tb.T)
    frac_down = torch.norm(ta, dim=-1).view(bs1, 1).repeat(1, bs2) * \
                torch.norm(tb, dim=-1).view(1, bs2).repeat(bs1, 1)
    return frac_up / frac_down



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def extract_inception_features(
    images: torch.Tensor,
    inception_model: nn.Module,
    batch_size: int = 1024,
    device: str = "cuda"
):
    """
    images: Tensor [N, 3, H, W], range [0,1]
    return: Tensor [N, 2048]
    """
    inception_model.eval()
    feats = []

    loader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory= True
    )

    with torch.no_grad():
        for x in tqdm(loader, desc="Extracting Inception Features"):
            x = x.to(device)
            f = inception_model(x)          # [B, 2048]
            feats.append(f.cpu())           # 放 CPU，省显存

    return torch.cat(feats, dim=0)



class InceptionV3FeatureExtractor(nn.Module):
    """
    Differentiable Inception-V3 feature extractor.
    Output: 2048-d features before classification head.
    """
    def __init__(self):
        super().__init__()

        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False
        )

        # 关闭 aux logits（兼容 torchvision 版本）
        inception.aux_logits = False
        inception.AuxLogits = None

        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 冻结 Inception 参数（但不阻断梯度图）
        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, x):
        """
        x: Tensor [B, C, H, W], range [0, 1]
        """
        # Inception-V3 expects 299x299
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299),
                              mode='bilinear', align_corners=False)

        # ImageNet normalization（可导）
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # [B, 2048]
        return x


class PublicDistillationDataset(Dataset):
    def __init__(self, selected_public_distillations, transform=None):
        """
        Args:
            selected_public_distillations:
                List[List[np.ndarray]], outer index = class label
            transform:
                torchvision transforms applied to image
        """
        self.samples = []
        self.labels = []
        self.transform = transform

        for class_idx, class_samples in enumerate(selected_public_distillations):
            for img in class_samples:
                self.samples.append(img)
                self.labels.append(class_idx)

        self.samples = np.asarray(self.samples) # [N,H,W,C]
        self.labels = np.asarray(self.labels) # [N]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]  # np.ndarray, uint8, HWC
        label = self.labels[idx]

        # numpy -> PIL Image
        img = Image.fromarray(img)

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback: ToTensor
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        return img, label




class DP_Diffusion(DPSynther):
    def __init__(self, config, device):
        """
        Initializes the model with the provided configuration and device settings.

        Args:
            config (Config): Configuration object containing all necessary parameters.
            device (str): Device to use for computation (e.g., 'cuda:0').
        """
        super().__init__()
        self.local_rank = config.local_rank  # Local rank of the process
        self.global_rank = config.global_rank  # Global rank of the process
        self.global_size = config.global_size  # Total number of processes

        self.denoiser_name = config.denoiser_name  # Name of the denoiser to be used
        self.denoiser_network = config.denoiser_network  # Network architecture for the denoiser
        self.ema_rate = config.ema_rate  # Rate for exponential moving average
        self.network = config.network  # Configuration for the network
        self.sampler = config.sampler  # Sampler configuration
        self.sampler_fid = config.sampler_fid  # FID sampler configuration
        self.sampler_acc = config.sampler_acc  # Accuracy sampler configuration
        self.fid_stats = config.fid_stats  # FID statistics configuration

        self.config = config  # Store the entire configuration
        self.device = 'cuda:%d' % self.local_rank  # Set the device based on local rank

        self.private_num_classes = config.private_num_classes  # Number of private classes
        self.public_num_classes = config.public_num_classes  # Number of public classes
        label_dim = max(self.private_num_classes, self.public_num_classes)  # Determine the maximum label dimension
        self.network.label_dim = label_dim  # Set the label dimension for the network
        self.network.label_dim = 1000

        # Initialize the denoiser based on the specified name and network
        if self.denoiser_name == 'edm':
            if self.denoiser_network == 'song':
                self.model = EDMDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize EDM denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for EDM denoiser")
        elif self.denoiser_name == 'vpsde':
            if self.denoiser_network == 'song':
                self.model = VPSDEDenoiser(self.config.beta_min, self.config.beta_max - self.config.beta_min,
                                        self.config.scale, NCSNpp(**self.network).to(self.device))  # Initialize VPSDE denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for VPSDE denoiser")
        elif self.denoiser_name == 'vesde':
            if self.denoiser_network == 'song':
                self.model = VESDEDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize VESDE denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for VESDE denoiser")
        elif self.denoiser_name == 'v':
            if self.denoiser_network == 'song':
                self.model = VDenoiser(NCSNpp(**self.network).to(self.device))  # Initialize V denoiser with NCSNpp network
            else:
                raise NotImplementedError("Network type not supported for V denoiser")
        else:
            raise NotImplementedError("Denoiser name not recognized")

        self.model = self.model.to(self.local_rank)  # Move the model to the specified device
        self.model.train()  # Set the model to training mode
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_rate)  # Initialize EMA for the model parameters

        # 可以对ckpt进行加载，直接加载state即可
        # Load checkpoint if provided
        if config.ckpt is not None:
            state = torch.load(config.ckpt, map_location=self.device)  # Load the checkpoint
            new_state_dict = {}
            for k, v in state['model'].items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v  # 去掉 "module."
                else:
                    new_state_dict[k] = v  # 原样保留
                # new_state_dict[k[7:]] = v  # Adjust the keys to match the model's state dictionary
            logging.info(self.model.load_state_dict(new_state_dict, strict=True))  # Load the state dictionary into the model
            self.ema.load_state_dict(state['ema'])  # Load the EMA state dictionary
            del state, new_state_dict  # Clean up memory


    def pretrain(self, public_dataloader, config):
        """
        Pre-trains the model using the provided public dataloader and configuration.

        Args:
            public_dataloader (DataLoader): The dataloader for the public dataset.
            config (dict): Configuration dictionary containing various settings and hyperparameters.
        """
        if public_dataloader is None or config.n_epochs == 0:
            # If no public dataloader is provided, return.
            return

        # Set the number of classes in the loss function to the number of private classes.
        config.loss.n_classes = self.public_num_classes
        if config.cond:
            # If conditional training is enabled, set the label unconditioning probability to 0.1. Conditional training with a low unconditioning probability usually performs better.
            config.loss['label_unconditioning_prob'] = 0.1
        else:
            # If conditional training is disabled, set the label unconditioning probability to 1.0.
            config.loss['label_unconditioning_prob'] = 1.0

        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # Wrap the model with DistributedDataParallel (DDP) for distributed training.
        model = DDP(self.model, device_ids=[self.local_rank])
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and training details if the global rank is 0.
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])
        dist.barrier()

        # Create a distributed data loader for the public dataset.
        dataset_loader = torch.utils.data.DataLoader(
            dataset=public_dataloader.dataset,
            batch_size=config.batch_size // self.global_size,
            sampler=DistributedSampler(public_dataloader.dataset),
            pin_memory=True,
            drop_last=True,
            num_workers=16 if config.batch_size // self.global_size > 100 else 0
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels,
                                self.network.image_size,
                                self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size,
                            self.network.num_in_channels,
                            self.network.image_size,
                            self.network.image_size)

        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):
            dataset_loader.sampler.set_epoch(epoch)
            for _, (train_x, train_y) in enumerate(dataset_loader):

                # Save snapshots and checkpoints at specified intervals.
                if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                    logging.info('Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                            sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes)
                        ema.restore(model.parameters())
                    model.train()

                    save_checkpoint(os.path.join(checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                dist.barrier()

                # Compute FID at specified intervals.
                if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes)
                        ema.restore(model.parameters())

                        if self.global_rank == 0:
                            logging.info('FID between synthetic images and sensitive images at iteration %d: %.6f' % (state['step'], fid))
                    model.train()
                dist.barrier()

                # Save checkpoints at specified intervals.
                if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info('Saving checkpoint at iteration %d' % state['step'])
                dist.barrier()

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)
                train_x, train_y = train_x.to(self.device) * 2. - 1., train_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(loss_fn(model, train_x, train_y))
                loss.backward()
                optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('Loss: %.4f, step: %d' % (loss, state['step'] + 1))
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def train(self, sensitive_dataloader, config):
        """
        Trains the model using the provided sensitive data loader and configuration.

        Args:
            sensitive_dataloader (DataLoader): DataLoader containing the sensitive data.
            config (Config): Configuration object containing various settings for training.

        Returns:
            None
        """
        if sensitive_dataloader is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank
        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes

        # Define directories for saving samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if this is the main process.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        if config.partly_finetune:
            # If partial fine-tuning is enabled, freeze certain layers.
            trainable_parameters = []
            for name, param in self.model.named_parameters():
                # layer_idx 取层数id， 层数小于3 和层数大于3层的NIN层进行微调
                layer_idx = int(name.split('.')[2])
                if layer_idx > 3 and 'NIN' not in name:
                    param.requires_grad = False
                    if self.global_rank == 0:
                        logging.info('{} is frozen'.format(name))
                else:
                    trainable_parameters.append(param)
        else:
            # Otherwise, all parameters are trainable.
            trainable_parameters = self.model.parameters()

        # 模型进行分布式差分隐私训练
        # Wrap the model with DPDDP for distributed training with differential privacy.
        model = DPDDP(self.model)
        # Initialize Exponential Moving Average (EMA) for model parameters.
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(trainable_parameters, **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the state dictionary to keep track of the training process.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        if self.global_rank == 0:
            # Log the number of trainable parameters and other training details.
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            n_params = sum([np.prod(p.size()) for p in model_parameters])
            logging.info('Number of trainable parameters in model: %d' % n_params)
            logging.info('Number of total epochs: %d' % config.n_epochs)
            logging.info('Starting training at step %d' % state['step'])

        # Initialize the Privacy Engine for differential privacy.
        privacy_engine = PrivacyEngine(accountant='rdp' if 'accountant' not in config else config.accountant)
        if config.dp.privacy_history is None:
            account_history = None
        else:
            account_history = [tuple(item) for item in config.dp.privacy_history]

        # Make the model, optimizer, and data loader private.
        model, optimizer, dataset_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=sensitive_dataloader,
            target_delta=config.dp.delta,
            target_epsilon=config.dp.epsilon,
            epochs=config.n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
            account_history=account_history,
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function not supported")

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shapes for sampling images.
        # snapshot_sampling_shape 默认大小是（80,3,32, 32）
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                self.network.num_in_channels, self.network.image_size, self.network.image_size)

        # fid_sampling_shape 默认大小是 （256, 3, 32,32）
        fid_sampling_shape = (self.sampler.fid_batch_size, self.network.num_in_channels,
                            self.network.image_size, self.network.image_size)

        # Start the training loop.
        for epoch in range(config.n_epochs):
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizer,
                    n_splits=config.n_splits if config.n_splits > 0 else None) as memory_safe_data_loader:

                for _, (train_x, train_y) in enumerate(memory_safe_data_loader):
                    if state['step'] % config.snapshot_freq == 0 and state['step'] >= config.snapshot_threshold and self.global_rank == 0:
                        # Save a snapshot checkpoint and sample a batch of images.
                        logging.info(
                            'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])

                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            # 采样并保存中间训练结果
                            sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                                sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes)
                            ema.restore(model.parameters())
                        model.train()

                        save_checkpoint(os.path.join(
                            checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                    dist.barrier()

                    # config.fid_freq 一般而言为2000，也就是每2000个step进行保存
                    if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                        # Compute FID score and log it.
                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            # fid_samples = 5000, global_size = 4, self.fid_stats 是提前计算好的fid
                            fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model, self.fid_stats, self.device, self.private_num_classes)
                            ema.restore(model.parameters())

                            if self.global_rank == 0:
                                logging.info('FID at iteration %d: %.6f' % (state['step'], fid))
                            dist.barrier()
                        model.train()

                    if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                        # Save a checkpoint at regular intervals.
                        checkpoint_file = os.path.join(
                            checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info(
                            'Saving checkpoint at iteration %d' % state['step'])
                    dist.barrier()

                    #
                    if len(train_y.shape) == 2:
                        # Preprocess the input data.
                        train_x = train_x.to(torch.float32) / 255.
                        train_y = torch.argmax(train_y, dim=1)

                    x = train_x.to(self.device) * 2. - 1.
                    y = train_y.to(self.device).long()

                    # Perform a forward pass and backpropagation.
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, y))
                    loss.backward()
                    optimizer.step()

                    if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                        # Log the loss at regular intervals.
                        logging.info('Loss: %.4f, step: %d' %
                                    (loss, state['step'] + 1))
                    dist.barrier()

                    state['step'] += 1
                    if not optimizer._is_last_step_skipped:
                        state['ema'].update(model.parameters())

                # Log the epsilon value after each epoch.
                if self.global_rank == 0:
                    logging.info('Eps-value after %d epochs: %.4f' % (epoch + 1, privacy_engine.get_epsilon(config.dp.delta)))

        if self.global_rank == 0:
            # Save the final checkpoint.
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            # save的checkpoint包含4个参数，model，ema,optimizer,step
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')
        dist.barrier()

        # Update the EMA.
        self.ema = ema


    def distillation_with_syn_dataset(self, syn_data, syn_label, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if syn_data is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)


        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        synthetic_images_train_torch = torch.from_numpy(syn_data).float()
        synthetic_labels_train_torch = torch.from_numpy(syn_label)

        # with torch.no_grad():
        #     feat_train_x = inception_model(synthetic_images_train_torch)

        print("Extracting Inception features (one-time)...")

        synthetic_features_train_torch = extract_inception_features(
            synthetic_images_train_torch,
            inception_v3_model,
            batch_size=1024,          # 可以开大一点
            device=self.device
        )
        torch.cuda.empty_cache()
        print("Feature shape:", synthetic_features_train_torch.shape)

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)


        syn_dataset = TensorDataset(
            synthetic_images_train_torch,
            synthetic_features_train_torch,
            synthetic_labels_train_torch
        )

        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            syn_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_distill
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")



        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                   self.network.num_in_channels,
                                   self.network.image_size,
                                   self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size,
                              self.network.num_in_channels,
                              self.network.image_size,
                              self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, (train_x, feat_train_x, train_y) in enumerate(dataset_loader):

                # # Save snapshots and checkpoints at specified intervals.
                # if state['step'] % config.snapshot_freq == 0 and state[
                #     'step'] >= config.snapshot_threshold and self.global_rank == 0:
                #     logging.info(
                #         'Saving snapshot checkpoint and sampling single batch at iteration %d.' % state['step'])
                #
                #     model.eval()
                #     with torch.no_grad():
                #         ema.store(model.parameters())
                #         ema.copy_to(model.parameters())
                #         sample_random_image_batch(snapshot_sampling_shape, sampler, os.path.join(
                #             sample_dir, 'iter_%d' % state['step']), self.device, self.private_num_classes)
                #         ema.restore(model.parameters())
                #     model.train()
                #
                #     save_checkpoint(os.path.join(checkpoint_dir, 'snapshot_checkpoint.pth'), state)
                #
                # # Compute FID at specified intervals.
                # if state['step'] % config.fid_freq == 0 and state['step'] >= config.fid_threshold:
                #     model.eval()
                #     with torch.no_grad():
                #         ema.store(model.parameters())
                #         ema.copy_to(model.parameters())
                #         fid = compute_fid(config.fid_samples, self.global_size, fid_sampling_shape, sampler, inception_model,
                #                           self.fid_stats, self.device, self.private_num_classes)
                #         ema.restore(model.parameters())
                #
                #         if self.global_rank == 0:
                #             logging.info(
                #                 'FID between synthetic images and sensitive images at iteration %d: %.6f' % (state['step'], fid))
                #     model.train()
                #
                #
                # # Save checkpoints at specified intervals.
                # if state['step'] % config.save_freq == 0 and state['step'] >= config.save_threshold and self.global_rank == 0:
                #     checkpoint_file = os.path.join(
                #         checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                #     save_checkpoint(checkpoint_file, state)
                #     logging.info('Saving checkpoint at iteration %d' % state['step'])


                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, feat_train_x, train_y = train_x.to(self.device), feat_train_x.to(self.device), train_y.to(self.device)


                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y, teacher)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)


                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)


                for c in ry_set:
                    mask = (train_y == c)
                    if len(pseudo_memory[c]):
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.008 / len(ry_set)

                    real_memory[c].extend(feat_train_x[mask].detach().split(1))
                    pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                    while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def fine_tune_with_syn_dataset(self, syn_data, syn_label, config):
        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if syn_data is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return


        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes

        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        synthetic_images_train_torch = torch.from_numpy(syn_data).float()
        synthetic_labels_train_torch = torch.from_numpy(syn_label)

        # with torch.no_grad():
        #     feat_train_x = inception_model(synthetic_images_train_torch)

        print("Extracting Inception features (one-time)...")

        synthetic_features_train_torch = extract_inception_features(
            synthetic_images_train_torch,
            inception_v3_model,
            batch_size=1024,  # 可以开大一点
            device=self.device
        )
        torch.cuda.empty_cache()
        print("Feature shape:", synthetic_features_train_torch.shape)

        # Initialize the Inception model for feature extraction.
        inception_model = InceptionFeatureExtractor()
        inception_model.model = inception_model.model.to(self.device)

        syn_dataset = TensorDataset(
            synthetic_images_train_torch,
            synthetic_features_train_torch,
            synthetic_labels_train_torch
        )

        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            syn_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_fine_tune
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)

            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Define the shape of the batches for sampling and FID computation.
        snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
                                   self.network.num_in_channels,
                                   self.network.image_size,
                                   self.network.image_size)
        fid_sampling_shape = (self.sampler.fid_batch_size,
                              self.network.num_in_channels,
                              self.network.image_size,
                              self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)

        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, (train_x, feat_train_x, train_y) in enumerate(dataset_loader):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, feat_train_x, train_y = train_x.to(self.device), feat_train_x.to(self.device), train_y.to(self.device)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)


                for c in ry_set:
                    mask = (train_y == c)
                    if len(pseudo_memory[c]):
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.008 / len(ry_set)

                    real_memory[c].extend(feat_train_x[mask].detach().split(1))
                    pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                    while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()

                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (
                    all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))

                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')

        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def distillation_with_public_dataset(self, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_distillations is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        # # Initialize the Inception model for feature extraction.
        # inception_model = InceptionFeatureExtractor()
        # inception_model.model = inception_model.model.to(self.device)

        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        public_distillation_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_distillations,
            transform=public_distillation_transform
        )


        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            public_distillation_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_distill
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, (train_x, train_y) in enumerate(dataset_loader):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                feat_train_x = inception_v3_model(train_x)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y, teacher)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)


                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)


                for c in ry_set:
                    mask = (train_y == c)
                    if len(pseudo_memory[c]):
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.004 / len(ry_set)

                    real_memory[c].extend(feat_train_x[mask].detach().split(1))
                    pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                    while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def distillation_with_public_dataset_advanced_only(self, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_prototypes is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        # # Initialize the Inception model for feature extraction.
        # inception_model = InceptionFeatureExtractor()
        # inception_model.model = inception_model.model.to(self.device)

        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        public_distillation_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_prototypes,
            transform=public_distillation_transform
        )


        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            public_distillation_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_distill
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, (train_x, train_y) in enumerate(dataset_loader):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                feat_train_x = inception_v3_model(train_x)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y, teacher)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)


                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)


                for c in ry_set:
                    mask = (train_y == c)
                    if len(pseudo_memory[c]):
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.008 / len(ry_set)

                    real_memory[c].extend(feat_train_x[mask].detach().split(1))
                    pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                    while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()





    def distillation_with_public_dataset_advanced(self, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_distillations is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        public_distillation_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_distillations,
            transform=public_distillation_transform
        )

        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            public_distillation_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        prototypes_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_prototypes,
            transform=public_distillation_transform
        )

        prototypes_loader = DataLoader(prototypes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_distill
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, ((train_x, train_y), (proto_x, proto_y)) in enumerate(zip(dataset_loader, cycle(prototypes_loader))):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                proto_x, proto_y = proto_x.to(self.device), proto_y.to(self.device)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y, teacher)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)

                feat_proto_x = inception_v3_model(proto_x)

                for c in ry_set:
                    mask = (train_y == c)
                    mask_proto = (proto_y == c)
                    if mask.any() and len(real_memory[c]) > 0 and len(pseudo_memory[c]) > 0:
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.008 / len(ry_set)

                    # 无论是否计算相似度，都要更新记忆
                    if mask_proto.any():
                        real_memory[c].extend(feat_proto_x[mask_proto].detach().split(1))
                    if mask.any():
                        pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def train_with_public_dataset(self, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_distillations is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        # # Initialize the Inception model for feature extraction.
        # inception_model = InceptionFeatureExtractor()
        # inception_model.model = inception_model.model.to(self.device)

        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        public_distillation_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_prototypes,
            transform=public_distillation_transform
        )


        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            public_distillation_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_fine_tune
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, (train_x, train_y) in enumerate(dataset_loader):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                feat_train_x = inception_v3_model(train_x)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)


                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)


                for c in ry_set:
                    mask = (train_y == c)
                    if len(pseudo_memory[c]):
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.008 / len(ry_set)

                    real_memory[c].extend(feat_train_x[mask].detach().split(1))
                    pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                    while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                    while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = 0.1 * loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def train_with_public_dataset_advanced(self, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_distillations is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        public_distillation_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_distillations,
            transform=public_distillation_transform
        )

        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            public_distillation_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        prototypes_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_prototypes,
            transform=public_distillation_transform
        )

        prototypes_loader = DataLoader(prototypes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_fine_tune
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, ((train_x, train_y), (proto_x, proto_y)) in enumerate(zip(dataset_loader, cycle(prototypes_loader))):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, train_y = train_x.to(self.device), train_y.to(self.device)
                proto_x, proto_y = proto_x.to(self.device), proto_y.to(self.device)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)

                feat_proto_x = inception_v3_model(proto_x)

                for c in ry_set:
                    mask = (train_y == c)
                    mask_proto = (proto_y == c)
                    if mask.any() and len(real_memory[c]) > 0 and len(pseudo_memory[c]) > 0:
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.002 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.008 / len(ry_set)

                    # 无论是否计算相似度，都要更新记忆
                    if mask_proto.any():
                        real_memory[c].extend(feat_proto_x[mask_proto].detach().split(1))
                    if mask.any():
                        pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = 0.01 * loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()


    def distill_syn_data_with_public_dataset_advanced(self, syn_data, syn_label, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_distillations is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        synthetic_images_train_torch = torch.from_numpy(syn_data).float()
        synthetic_labels_train_torch = torch.from_numpy(syn_label)

        print("Extracting Inception features (one-time)...")

        synthetic_features_train_torch = extract_inception_features(
            synthetic_images_train_torch,
            inception_v3_model,
            batch_size=1024,  # 可以开大一点
            device=self.device
        )
        torch.cuda.empty_cache()
        print("Feature shape:", synthetic_features_train_torch.shape)

        syn_dataset = TensorDataset(
            synthetic_images_train_torch,
            synthetic_features_train_torch,
            synthetic_labels_train_torch
        )


        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        #
        # public_distillation_dataset = PublicDistillationDataset(
        #     selected_public_distillations=selected_public_distillations,
        #     transform=public_distillation_transform
        # )

        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            syn_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        prototypes_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_prototypes,
            transform=public_distillation_transform
        )

        prototypes_loader = DataLoader(prototypes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_distill
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, ((train_x, feat_train_x, train_y), (proto_x, proto_y)) in enumerate(zip(dataset_loader, cycle(prototypes_loader))):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, feat_train_x, train_y = train_x.to(self.device), feat_train_x.to(self.device), train_y.to(self.device)
                proto_x, proto_y = proto_x.to(self.device), proto_y.to(self.device)

                train_x = train_x * 2. - 1.

                loss, pred_student = loss_fn(model, train_x, train_y, teacher)
                pred_student = pred_student.mean(dim=1)
                loss = torch.mean(loss)

                pred_student = (pred_student + 1.) / 2
                pred_student = pred_student.clamp(0, 1)

                feat_pred_student = inception_v3_model(pred_student)

                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)

                feat_proto_x = inception_v3_model(proto_x)

                for c in ry_set:
                    mask = (train_y == c)
                    mask_proto = (proto_y == c)
                    if mask.any() and len(real_memory[c]) > 0 and len(pseudo_memory[c]) > 0:
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.1 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.1 / len(ry_set)

                    # 无论是否计算相似度，都要更新记忆
                    if mask_proto.any():
                        real_memory[c].extend(feat_proto_x[mask_proto].detach().split(1))
                    if mask.any():
                        pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()

    def distill_syn_data_with_public_prototype(self, syn_data, syn_label, selected_public_prototypes, selected_public_distillations, config):

        self.global_rank = 0
        self.global_size = 1
        self.local_rank = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(0)

        if selected_public_distillations is None or config.n_epochs == 0:
            # If the dataloader is not provided or the number of epochs is zero, exit early.
            return

        set_seeds(self.global_rank, config.seed)
        # Set the CUDA device based on the local rank.
        torch.cuda.device(self.local_rank)
        self.device = 'cuda:%d' % self.local_rank

        # Set the number of classes for the loss function.
        config.loss.n_classes = self.private_num_classes


        # Define directories for storing samples and checkpoints.
        sample_dir = os.path.join(config.log_dir, 'samples')
        checkpoint_dir = os.path.join(config.log_dir, 'checkpoints')

        if self.global_rank == 0:
            # Create necessary directories if the global rank is 0.
            make_dir(config.log_dir)
            make_dir(sample_dir)
            make_dir(checkpoint_dir)

        # 改为单卡蒸馏
        # teacher model 冻结
        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # student model 可训练
        model = self.model
        model.train()
        for p in model.parameters():
            p.requires_grad = True


        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)

        # Initialize the optimizer based on the configuration.
        if config.optim.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
        elif config.optim.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **config.optim.params)
        else:
            raise NotImplementedError("Optimizer not supported")

        # Initialize the training state.
        state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

        # 构建inception_V3_model
        inception_v3_model = InceptionV3FeatureExtractor().to(self.device)

        synthetic_images_train_torch = torch.from_numpy(syn_data).float()
        synthetic_labels_train_torch = torch.from_numpy(syn_label)

        # print("Extracting Inception features (one-time)...")
        #
        # synthetic_features_train_torch = extract_inception_features(
        #     synthetic_images_train_torch,
        #     inception_v3_model,
        #     batch_size=1024,  # 可以开大一点
        #     device=self.device
        # )
        # torch.cuda.empty_cache()
        # print("Feature shape:", synthetic_features_train_torch.shape)

        # syn_dataset = TensorDataset(
        #     synthetic_images_train_torch,
        #     synthetic_features_train_torch,
        #     synthetic_labels_train_torch
        # )

        syn_dataset = TensorDataset(
            synthetic_images_train_torch,
            synthetic_labels_train_torch
        )

        # 构造public_distllations的transform
        public_distillation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

        # public_distillation_dataset = PublicDistillationDataset(
        #     selected_public_distillations=selected_public_prototypes,
        #     transform=public_distillation_transform
        # )

        # 构建单卡场景下的dataset_loader
        dataset_loader = DataLoader(
            syn_dataset,
            batch_size= config.batch_size,
            shuffle=True,
            num_workers = 4,
            pin_memory = True,
            )

        prototypes_dataset = PublicDistillationDataset(
            selected_public_distillations=selected_public_prototypes,
            transform=public_distillation_transform
        )

        prototypes_loader = DataLoader(prototypes_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)


        # Initialize the loss function based on the configuration.
        if config.loss.version == 'edm':
            loss_fn = EDMLoss(**config.loss).get_loss_distill
        elif config.loss.version == 'vpsde':
            loss_fn = VPSDELoss(**config.loss).get_loss
        elif config.loss.version == 'vesde':
            loss_fn = VESDELoss(**config.loss).get_loss
        elif config.loss.version == 'v':
            loss_fn = VLoss(**config.loss).get_loss
        else:
            raise NotImplementedError("Loss function version not supported")


        # Define the sampler function for generating images.
        def sampler(x, y=None):
            if self.sampler.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler)
            elif self.sampler.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler)
            else:
                raise NotImplementedError("Sampler type not supported")

        # # Define the shape of the batches for sampling and FID computation.
        # snapshot_sampling_shape = (self.sampler.snapshot_batch_size,
        #                            self.network.num_in_channels,
        #                            self.network.image_size,
        #                            self.network.image_size)
        # fid_sampling_shape = (self.sampler.fid_batch_size,
        #                       self.network.num_in_channels,
        #                       self.network.image_size,
        #                       self.network.image_size)

        real_memory = defaultdict(list)
        pseudo_memory = defaultdict(list)


        # Training loop over the specified number of epochs.
        for epoch in range(config.n_epochs):

            for _, ((train_x, train_y), (proto_x, proto_y)) in enumerate(zip(dataset_loader, cycle(prototypes_loader))):

                # Prepare the input data for training.
                if len(train_y.shape) == 2:
                    train_x = train_x.to(torch.float32) / 255.
                    train_y = torch.argmax(train_y, dim=1)

                train_x, train_y = train_x.to(self.device), train_y.to(self.device)

                train_x = train_x * 2. - 1.

                loss_train, pred_student_train = loss_fn(model, train_x, train_y, teacher)

                # Prepare the input data for training.
                if len(proto_y.shape) == 2:
                    proto_x = proto_x.to(torch.float32) / 255.
                    proto_y = torch.argmax(proto_y, dim=1)

                proto_x, proto_y = proto_x.to(self.device), proto_y.to(self.device)

                proto_x = proto_x * 2. - 1.

                loss_proto, pred_student_proto = loss_fn(model, proto_x, proto_y, teacher)

                pred_student_train = pred_student_train.mean(dim=1)
                loss_train = torch.mean(loss_train)

                pred_student_proto = pred_student_proto.mean(dim=1)
                loss_proto = torch.mean(loss_proto)


                pred_student_train = (pred_student_train + 1.) / 2
                pred_student_train = pred_student_train.clamp(0, 1)

                feat_pred_student_train = inception_v3_model(pred_student_train)

                pred_student_proto = (pred_student_proto + 1.) / 2
                pred_student_proto = pred_student_proto.clamp(0, 1)

                feat_pred_student_proto = inception_v3_model(pred_student_proto)



                pos_match_loss, neg_match_loss = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)


                merged_y = torch.cat([train_y, proto_y], dim=0)


                ry_set = train_y.unique().tolist()
                # ry_set = set(train_y)

                feat_proto_x = inception_v3_model(proto_x)

                for c in ry_set:
                    mask = (train_y == c)
                    mask_proto = (proto_y == c)
                    if mask.any() and len(real_memory[c]) > 0 and len(pseudo_memory[c]) > 0:
                        pos_embeddings = torch.cat(real_memory[c])
                        neg_embeddings = torch.cat(pseudo_memory[c])

                        # 表示性限制
                        pos_feat_sim = 1 - cosine_similarity(feat_pred_student[mask], pos_embeddings).min()

                        # 多样性限制
                        neg_feat_sim = cosine_similarity(feat_pred_student[mask], neg_embeddings).max()

                        pos_match_loss += pos_feat_sim * 0.1 / len(ry_set)
                        neg_match_loss += neg_feat_sim * 0.1 / len(ry_set)

                    # 无论是否计算相似度，都要更新记忆
                    if mask_proto.any():
                        real_memory[c].extend(feat_proto_x[mask_proto].detach().split(1))
                    if mask.any():
                        pseudo_memory[c].extend(feat_pred_student[mask].detach().split(1))

                while len(real_memory[c]) > 64:
                        real_memory[c].pop(0)
                while len(pseudo_memory[c]) > 64:
                        pseudo_memory[c].pop(0)


                all_loss = loss + pos_match_loss + neg_match_loss
                if state['step'] != 0:
                    optimizer.zero_grad(set_to_none=True)
                    all_loss.backward()
                    optimizer.step()


                # Log the loss at specified intervals.
                if (state['step'] + 1) % config.log_freq == 0 and self.global_rank == 0:
                    logging.info('all_loss: %.4f, loss: %.4f, pos_match_loss: %.4f, neg_match_loss: %.4f, step: %d' % (all_loss, loss, pos_match_loss, neg_match_loss, state['step'] + 1))


                state['step'] += 1
                state['ema'].update(model.parameters())
            if self.global_rank == 0:
                logging.info('Completed Epoch %d' % (epoch + 1))

        # Save the final checkpoint.
        if self.global_rank == 0:
            checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
            save_checkpoint(checkpoint_file, state)
            logging.info('Saving final checkpoint.')


        # Apply the EMA weights to the model and store the EMA object.
        ema.copy_to(self.model.parameters())
        self.model = model
        self.ema = ema

        # Clean up the model and free GPU memory.
        del model
        torch.cuda.empty_cache()




    # generate 部分代码应该完全不用调整
    def generate(self, config):
        # Log the start of the generation process with the number of samples to be generated
        logging.info("start to generate {} samples".format(config.data_num))

        # Ensure the log directory exists if this is the main process (global_rank == 0)
        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)

        # Synchronize all processes
        dist.barrier()

        # Define the shape for the sampling batch
        sampling_shape = (config.batch_size, self.network.num_in_channels, self.network.image_size, self.network.image_size)

        # Wrap the model with DistributedDataParallel for multi-GPU training
        model = DDP(self.model)
        model.eval()  # Set the model to evaluation mode

        # Copy the exponential moving average parameters to the model
        self.ema.copy_to(model.parameters())

        # Define a function to handle different types of samplers
        def sampler_acc(x, y=None):
            if self.sampler_acc.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler_acc)
            elif self.sampler_acc.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler_acc)
            else:
                raise NotImplementedError("Sampler type not supported")

        # Initialize lists to store synthetic data and labels if this is the main process
        if self.global_rank == 0:
            syn_data = []
            syn_labels = []

        # Loop to generate the required number of samples
        for _ in range(config.data_num // (sampling_shape[0] * self.global_size) + 1):
            # Generate a batch of samples and labels
            x, y = generate_batch(sampler_acc, sampling_shape, self.device, self.private_num_classes, self.private_num_classes)

            # Synchronize all processes
            dist.barrier()

            # Prepare tensors for gathering results from all processes
            if self.global_rank == 0:
                gather_x = [torch.zeros_like(x) for _ in range(self.global_size)]
                gather_y = [torch.zeros_like(y) for _ in range(self.global_size)]
            else:
                gather_x = None
                gather_y = None

            # Gather the generated samples and labels from all processes
            dist.gather(x, gather_x)
            dist.gather(y, gather_y)

            # If this is the main process, collect the gathered data
            if self.global_rank == 0:
                syn_data.append(torch.cat(gather_x).detach().cpu().numpy())
                syn_labels.append(torch.cat(gather_y).detach().cpu().numpy())

        # If this is the main process, finalize the generation process
        if self.global_rank == 0:
            logging.info("Generation Finished!")

            # Concatenate all collected synthetic data and labels
            syn_data = np.concatenate(syn_data)[:config.data_num]
            syn_labels = np.concatenate(syn_labels)[:config.data_num]

            # Save the synthetic data and labels to a .npz file
            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

            # Prepare images to display, 依旧准备一个8*num_classes的图像
            show_images = []
            for cls in range(self.private_num_classes):
                show_images.append(syn_data[syn_labels==cls][:8])
            show_images = np.concatenate(show_images)

            # Save the sample images to a PNG file
            torchvision.utils.save_image(torch.from_numpy(show_images), os.path.join(config.log_dir, 'sample.png'), padding=1, nrow=8)

            # Return the synthetic data and labels
            return syn_data, syn_labels
        else:
            # Return None for non-main processes
            return None, None


    def generate_normal(self, config):
        logging.info(f"start to generate {config.data_num} samples")

        # ===== 判断是否为分布式 =====
        is_distributed = dist.is_available() and dist.is_initialized()

        if not is_distributed:
            # 单卡兜底
            self.global_rank = 0
            self.global_size = 1

        if self.global_rank == 0 and not os.path.exists(config.log_dir):
            make_dir(config.log_dir)

        if is_distributed:
            dist.barrier()

        sampling_shape = (
            config.batch_size,
            self.network.num_in_channels,
            self.network.image_size,
            self.network.image_size,
        )

        # ===== Model / EMA =====
        model = self.model
        model.eval()
        self.ema.copy_to(model.parameters())

        if is_distributed:
            model = DDP(model)

        def sampler_acc(x, y=None):
            if self.sampler_acc.type == 'ddim':
                return ddim_sampler(x, y, model, **self.sampler_acc)
            elif self.sampler_acc.type == 'edm':
                return edm_sampler(x, y, model, **self.sampler_acc)
            else:
                raise NotImplementedError

        if self.global_rank == 0:
            syn_data, syn_labels = [], []

        n_rounds = config.data_num // (sampling_shape[0] * self.global_size) + 1

        for _ in range(n_rounds):
            x, y = generate_batch(
                sampler_acc,
                sampling_shape,
                self.device,
                self.private_num_classes,
                self.private_num_classes,
            )

            if is_distributed:
                dist.barrier()

                if self.global_rank == 0:
                    gather_x = [torch.zeros_like(x) for _ in range(self.global_size)]
                    gather_y = [torch.zeros_like(y) for _ in range(self.global_size)]
                else:
                    gather_x = None
                    gather_y = None

                dist.gather(x, gather_x)
                dist.gather(y, gather_y)

                if self.global_rank == 0:
                    syn_data.append(torch.cat(gather_x).cpu().numpy())
                    syn_labels.append(torch.cat(gather_y).cpu().numpy())

            else:
                # ===== 单卡：直接收集 =====
                syn_data.append(x.detach().cpu().numpy())
                syn_labels.append(y.detach().cpu().numpy())

        if self.global_rank == 0:
            logging.info("Generation Finished!")

            syn_data = np.concatenate(syn_data)[:config.data_num]
            syn_labels = np.concatenate(syn_labels)[:config.data_num]

            np.savez(os.path.join(config.log_dir, "gen.npz"), x=syn_data, y=syn_labels)

            show_images = []
            for cls in range(self.private_num_classes):
                show_images.append(syn_data[syn_labels == cls][:8])
            show_images = np.concatenate(show_images)

            torchvision.utils.save_image(
                torch.from_numpy(show_images),
                os.path.join(config.log_dir, 'sample.png'),
                padding=1,
                nrow=8,
            )

            return syn_data, syn_labels

        return None, None