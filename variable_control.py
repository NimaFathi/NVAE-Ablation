# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
from PIL import Image
import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from time import time
import torch.distributed as dist

from torch.multiprocessing import Process
from torch.cuda.amp import autocast

from model import AutoEncoder
import utils
from distributions import Normal


def tile_image(batch_image):
    n = batch_image.size(0)
    channels, height, width = batch_image.size(1), batch_image.size(2), batch_image.size(3)
    batch_image = batch_image.permute(1, 0, 2, 3)

    batch_image = batch_image.reshape(channels, n * height, width)
    print(batch_image.shape)
    return batch_image


def create_hierarchichal_samples(model, fixed_z, t, z_list):
    z_count = 0
    scale_ind = 0
    if z_count < fixed_z:
        z = z_list[z_count]
    else:
        z0_size = [1] + model.z0_size
        dist = Normal(mu=torch.zeros(z0_size).cuda(), log_sigma=torch.zeros(z0_size).cuda(), temp=t)
        z, _ = dist.sample()
    idx_dec = 0
    s = model.prior_ftr0.unsqueeze(0)
    batch_size = z.size(0)
    s = s.expand(batch_size, -1, -1, -1)
    for cell in model.dec_tower:
        if cell.cell_type == 'combiner_dec':
            if idx_dec > 0:
                z_count += 1
                if z_count < fixed_z:
                    z = z_list[z_count]
                else:

                    # form prior
                    param = model.dec_sampler[idx_dec - 1](s)
                    mu, log_sigma = torch.chunk(param, 2, dim=1)
                    dist = Normal(mu, log_sigma, t)
                    z, _ = dist.sample()

            # 'combiner_dec'
            s = cell(s, z)
            idx_dec += 1
        else:
            s = cell(s)
            if cell.cell_type == 'up_dec':
                scale_ind += 1

    if model.vanilla_vae:
        s = model.stem_decoder(z)

    for cell in model.post_process:
        s = cell(s)

    logits = model.image_conditional(s)
    return logits


def sample(model, num_samples, t):
    z_list = []
    scale_ind = 0
    z0_size = [num_samples] + model.z0_size
    dist = Normal(mu=torch.zeros(z0_size).cuda(), log_sigma=torch.zeros(z0_size).cuda(), temp=t)
    z, _ = dist.sample()
    z_list.append(z)
    idx_dec = 0
    s = model.prior_ftr0.unsqueeze(0)
    batch_size = z.size(0)
    s = s.expand(batch_size, -1, -1, -1)
    for cell in model.dec_tower:
        if cell.cell_type == 'combiner_dec':
            if idx_dec > 0:
                # form prior
                param = model.dec_sampler[idx_dec - 1](s)
                mu, log_sigma = torch.chunk(param, 2, dim=1)
                dist = Normal(mu, log_sigma, t)
                z, _ = dist.sample()
                z_list.append(z)

            # 'combiner_dec'
            s = cell(s, z)
            idx_dec += 1
        else:
            s = cell(s)
            if cell.cell_type == 'up_dec':
                scale_ind += 1

    if model.vanilla_vae:
        s = model.stem_decoder(z)

    for cell in model.post_process:
        s = cell(s)

    logits = model.image_conditional(s)
    return logits, z_list


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6022'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    cometml_logger = None
    args_fields = {k: v for k, v in vars(args).items()}

    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()
        with autocast():
            for i in range(iter):
                if i % 10 == 0:
                    print('setting BN statistics iter %d out of %d' % (i + 1, iter))
                model.sample(num_samples, t)
        model.eval()


def main(eval_args):
    # ensures that weight initializations are all the same
    logging = utils.Logger(eval_args.local_rank, eval_args.save)

    # load a checkpoint
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        logging.info('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    if eval_args.batch_size > 0:
        args.batch_size = eval_args.batch_size

    logging.info('loaded the model at epoch %d', checkpoint['epoch'])
    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('num conv layers: %d', len(model.all_conv_layers))
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))

    bn_eval_mode = not eval_args.readjust_bn
    total_samples = 20 // eval_args.world_size  # num images per gpu
    num_samples = 1  # sampling batch size
    num_iter = int(np.ceil(total_samples / num_samples))  # num iterations per gpu

    with torch.no_grad():
        n = int(np.floor(np.sqrt(num_samples)))
        set_bn(model, bn_eval_mode, num_samples=num_samples, t=eval_args.temp, iter=500)
        for ind in range(num_iter):  # sampling is repeated.
            torch.cuda.synchronize()
            start = time()
            with autocast():
                logits, z_list = sample(model, num_samples, eval_args.temp)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                else output.sample()

            torch.cuda.synchronize()

            output_tiled = tile_image(output_img).cpu().numpy().transpose(1, 2, 0)
            output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
            output_tiled = np.squeeze(output_tiled)

            im = Image.fromarray(output_tiled)
            im.save(os.path.join(eval_args.save, 'CELEBA256_3/samples_%d_temp_%d_0.png' % (
                ind, eval_args.temp)))

            with autocast():
                for i in range(len(z_list)):
                    print(z_list[i].shape)

                    new_logits = create_hierarchichal_samples(model, i + 1, eval_args.temp, z_list)
                    if i == 0:
                        x = new_logits
                    logits = torch.cat((logits, new_logits), dim=0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                else output.sample()
            torch.cuda.synchronize()

            end = time()
            logging.info('sampling time per batch: %0.3f sec', (end - start))

            output_tiled = tile_image(output_img).cpu().numpy().transpose(1, 2, 0)
            output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
            output_tiled = np.squeeze(output_tiled)

            im = Image.fromarray(output_tiled)
            im.save(os.path.join(eval_args.save, 'CELEBA256_3/samples_%d_temp_%d_1.png' % (
                ind, eval_args.temp)))

            output = model.decoder_output(x)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                else output.sample()
            torch.cuda.synchronize()

            end = time()
            logging.info('sampling time per batch: %0.3f sec', (end - start))

            output_tiled = tile_image(output_img).cpu().numpy().transpose(1, 2, 0)
            output_tiled = np.asarray(output_tiled * 255, dtype=np.uint8)
            output_tiled = np.squeeze(output_tiled)

            im = Image.fromarray(output_tiled)
            im.save(os.path.join(eval_args.save, 'CELEBA256_3/samples_%d_temp_%d_2.png' % (
                ind, eval_args.temp)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results

    parser.add_argument('--checkpoint', type=str, default='/tmp/expr/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--save', type=str, default='/usr/local/data/nimafh/NVAE-Ablation/imgs/',
                        help='location of the checkpoint')
    parser.add_argument('--eval_mode', type=str, default='sample', choices=['sample', 'evaluate', 'evaluate_fid'],
                        help='evaluation mode. you can choose between sample or evaluate.')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--readjust_bn', action='store_true', default=False,
                        help='adding this flag will enable readjusting BN statistics.')
    parser.add_argument('--temp', type=float, default=0.7,
                        help='The temperature used for sampling.')
    parser.add_argument('--num_iw_samples', type=int, default=1000,
                        help='The number of IW samples used in test_ll mode.')
    parser.add_argument('--fid_dir', type=str, default='/tmp/fid-stats',
                        help='path to directory where fid related files are stored')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size used during evaluation. If set to zero, training batch size is used.')
    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')

    args = parser.parse_args()
    utils.create_exp_dir(args.save)

    size = args.world_size

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            p = Process(target=init_processes, args=(rank, size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)