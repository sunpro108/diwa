import os
import argparse
import logging

import torch

import data as Data
import model as Model
import core.logger as Logger
import core.metrics as Metrics

if __name__ == "__main__":
    # region: parse configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    # endregion

    # region: logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    # endregion

    # region: dataset loader initialization
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    # endregion

    # region: create model and setup
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    
    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    
    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))
    
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    # endregion

    # region: training
    while current_step < n_iter:
        current_epoch += 1
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    #tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0
                result_path = '{}/{}'.format(opt['path']
                                                ['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['val'], schedule_phase='val')
                for _,  val_data in enumerate(val_loader):
                    idx += 1
                    diffusion.feed_data(val_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                    lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                    fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                    # generation
                    Metrics.save_img(
                        hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(
                        sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(
                        lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                    Metrics.save_img(
                        fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                    avg_psnr += Metrics.calculate_psnr(
                        sr_img, hr_img)
                    avg_ssim += Metrics.calculate_ssim(sr_img, hr_img)

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                diffusion.set_new_noise_schedule(
                    opt['model']['beta_schedule']['train'], schedule_phase='train')
                # log
                logger.info('# Validation # PSNR: {:.4e}, SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
                    current_epoch, current_step, avg_psnr, avg_ssim))

            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

    # save model
    logger.info('End of training.')
    # endregion