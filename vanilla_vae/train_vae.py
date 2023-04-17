# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import torch
import torch.nn as nn
import numpy as np
import os

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from vanilla_vae.model import VariationalAutoencoder, loss_function
import utils
import datasets

from comet import CometML


def main(args, comet_logger=None):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    # train_queue = torch.utils.data.Subset(train_queue, list(range(1, len(train_queue), 10)))

    model = VariationalAutoencoder()
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))

    cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                       weight_decay=1e-2, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - 4 - 1), eta_min=1e-4)
    grad_scalar = GradScaler(2 ** 10)

    args.k = 1
    args.increase = False
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')

    global_step, init_epoch = 0, 0
    if comet_logger is not None:
        experiment = comet_logger.get_experiment()

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > 4:
            cnn_scheduler.step()
        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, global_step = train(train_queue, model, cnn_optimizer, grad_scalar, global_step, epoch,
                                         4, writer, logging, comet_experiment=experiment, args=args)
        logging.info('train_nelbo %f', train_nelbo)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)

        save_freq = int(np.ceil(args.epochs / 100))

        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                            'args': args, 'scheduler': cnn_scheduler.state_dict(),
                            'grad_scalar': grad_scalar.state_dict()}, checkpoint_file)


def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, epoch, warmup_iters, writer, logging,
          comet_experiment, args):
    nelbo = utils.AvgrageMeter()
    model.train()
    total_len = len(train_queue)
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()
        # change bit length
        x = utils.pre_process(x, 5)
        batch_size = x.size(0)
        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)
        if step % 50 == 0:
            if args.increase:
                if args.k > 5:
                    args.increase = False
                    print('decrease')
                else:
                    args.k = args.k * 1.05

            elif not args.increase:
                if args.k < 0.2:
                    args.increase = True
                    print('increase')
                else:
                    args.k = args.k * 0.95

        cnn_optimizer.zero_grad()
        # with autocast():
        output = model(x)
        loss = loss_function(x, args.k, output[0], output[1], output[2])

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)
        if (global_step + 1) % 10 == 0:
            if (global_step + 1) % 50 == 0:  # reduced frequency
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = x[:n * n]
                output_img = output[0][:n * n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, global_step)
                rec_imgs = in_out_tiled.cpu().detach().numpy().transpose(1, 2, 0)
                rec_imgs = np.asarray(rec_imgs * 255, dtype=np.uint8)
                rec_imgs = np.squeeze(rec_imgs)
                from PIL import Image
                im = Image.fromarray(rec_imgs)

                comet_experiment.log_image(im, name=f"reconstruction-step: {global_step}.jpeg", step=global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train epoch: %d-[%d/%d] \t globalstep: %d \t nelbo: %f \t loss: %f', epoch, step + 1,
                         total_len, global_step + 1, nelbo.avg, loss)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/recon_iter',
                              torch.mean(loss_function(x, args.k, output[0], output[1], output[2])), global_step)

            if comet_experiment is not None:
                comet_experiment.log_metric('train/nelbo_avg', nelbo.avg, step=global_step, epoch=epoch)
                comet_experiment.log_metric('train/k_coef', args.k, step=global_step, epoch=epoch)
                comet_experiment.log_metric('train/nelbo_bpd',
                                            nelbo.avg * (1. / np.log(2.) / utils.num_output(args.dataset)),
                                            step=global_step, epoch=epoch)
                comet_experiment.log_metric('train/nelbo_iter', loss, step=global_step, epoch=epoch)
                comet_experiment.log_metric('train/recon_iter',
                                            torch.mean(loss_function(x, args.k, output[0], output[1], output[2])),
                                            step=global_step, epoch=epoch)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test_reconstruction(valid_queue, model, num_samples, args):
    from PIL import Image
    if args.distributed:
        dist.barrier()
    model.eval()
    print("valid queue length", len(valid_queue))
    for step, x in enumerate(valid_queue):
        print(f"valid step {step}")
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, 5)
        with torch.no_grad():
            for k in range(num_samples):
                output = model(x)
                output_img = output[0]
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = x[:n * n]
                output_img = output_img[:n * n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                rec_imgs = in_out_tiled.cpu().detach().numpy().transpose(1, 2, 0)
                rec_imgs = np.asarray(rec_imgs * 255, dtype=np.uint8)
                rec_imgs = np.squeeze(rec_imgs)
                im = Image.fromarray(rec_imgs)

                im.save(os.path.join(args.save, 'reconstructed/vanillavae_step_%d_samples_%d.png' % (step, k)))


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6022'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    cometml_logger = None
    args_fields = {k: v for k, v in vars(args).items()}
    cometml_logger = CometML(api_key=args.comet_api_key, disabled=args.disable_comet,
                             project_name=args.comet_project_name, workspace=args.comet_workspace,
                             parameters=args_fields)

    fn(args, cometml_logger)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument("--id", type=int, required=True, help="an unique number for identifying the experiment")

    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                 'lsun_church_128', 'lsun_church_64'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')

    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')

    # Comet
    parser.add_argument('--comet_api_key', type=str, default=None, help='comet api key')
    parser.add_argument('--comet_project_name', type=str, default="NVAE", help='comet project name')
    parser.add_argument('--comet_workspace', type=str, default="nimafh", help='comet workspace')
    parser.add_argument('--disable_comet', action='store_true', default=False)

    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)


