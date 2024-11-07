import os
import random
import logging
import argparse
import torch
from torch import nn
from tqdm import tqdm
import scipy.io as scio
from models import deeplabv3plus
from datasets.data_loader_voc import *
from utils import metrics
from utils import ext_transforms as et
from utils import poly_lr_scheduler


GLOBAL_MESSAGE = f''


np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random-seed', type=int, default=1, help='default: 1')
    parser.add_argument('--checkpoint-save-prefix', type=str, default='./checkpoints', help='default: ./checkpoints')
    parser.add_argument('--log-save-prefix', type=str, default='./logs', help='default: ./logs')

    # Datset Options
    parser.add_argument('--voc-root', type=str, default='../data/VOC2012_AUG', help='default: ../data/VOC2012_AUG')
    parser.add_argument('--num-classes', type=int, default=21, help='default: 21')

    # Deeplab Options
    available_models = [name for name in deeplabv3plus.__dict__ if name.startswith('deeplabv3')]
    parser.add_argument('--model', type=str, default='deeplabv3plus_mobilenet_v3_large', choices=available_models, help='default: deeplabv3plus_mobilenet_v3_large')
    parser.add_argument('--output-stride', type=int, default=16, choices=[8, 16], help='default: 16')

    # Train Options
    parser.add_argument('--pretrained-backbone', action='store_false', default=True, help='default: True')
    parser.add_argument('--weights_backbone', type=str, help='default: None')
    parser.add_argument('--crop-size', type=int, default=513, help='default: 513')
    parser.add_argument('--num-epochs', type=int, default=100, help='default: 100')
    parser.add_argument('--batch-size', type=int, default=32, help='default: 32')
    # parser.add_argument("--aux-loss", action="store_true", default=False, help='default: False')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='default: 1e-4')
    parser.add_argument('--base-lr', type=float, default=0.01, help='default: 0.01')
    parser.add_argument('--poly-lr-power', type=float, default=0.9, help='default: 0.9')

    parser.add_argument('--val-only', action='store_true', default=False, help='default: False')
    parser.add_argument('--checkpoint-path', type=str, help='default: None')

    return parser


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def validate(model, val_iter, device, metric):
    metric.reset()
    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(val_iter, 'Validating', unit='img'):
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(features)['out']
            metric.add_batch(labels.cpu().numpy(), outputs.argmax(axis=1).cpu().numpy())


def name_new_dir(prefix, dir_name):
    if os.path.exists(os.path.join(prefix, dir_name)):
        dirs = sorted([dir for dir in os.listdir(prefix) if dir.startswith(dir_name + '_')])
        dir_name += '_' + f'{len(dirs)}'
    return os.path.join(prefix, dir_name)


def get_logger(name, level, formatter, file=None, console=False):
    if not (file or console):
        console = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if file:
        fh = logging.FileHandler(file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def main():
    opts = get_argparser().parse_args()

    seed_everything(opts.random_seed)

    model = deeplabv3plus.__dict__[opts.model](
        weights_backbone=opts.weights_backbone,
        pretrained_backbone=opts.pretrained_backbone,
        num_classes=opts.num_classes,
        # aux_loss=opts.aux_loss,
        output_stride=opts.output_stride,
    )

    # logging and checkpoint
    log_save_dir = name_new_dir(opts.log_save_prefix, f'{opts.model}_os{opts.output_stride}')
    os.makedirs(log_save_dir)

    logger_log = get_logger(
        'log', 
        logging.INFO, 
        logging.Formatter('%(asctime)s: %(message)s'), 
        os.path.join(log_save_dir, 'log.txt'), 
        True,
    )

    logger_log.info(GLOBAL_MESSAGE)

    logger_log.info(
        f'random_seed {opts.random_seed}'
        f', log_save_dir {log_save_dir}'
        f', voc_root {opts.voc_root}'
        f', model {opts.model}'
        f', output_stride {opts.output_stride}'
    )

    if not opts.val_only:
        checkpoint_save_dir = name_new_dir(opts.checkpoint_save_prefix, f'{opts.model}_os{opts.output_stride}')
        os.makedirs(checkpoint_save_dir)

        formatter = logging.Formatter('%(message)s')
        logger_loss = get_logger('loss', logging.INFO, formatter, os.path.join(log_save_dir, 'loss.txt'))
        logger_pa = get_logger('pa', logging.INFO, formatter, os.path.join(log_save_dir, 'PA.txt'))
        logger_cpa = get_logger('cpa', logging.INFO, formatter, os.path.join(log_save_dir, 'cPA.txt'))
        logger_mpa = get_logger('mpa', logging.INFO, formatter, os.path.join(log_save_dir, 'mPA.txt'))
        logger_iou =  get_logger('iou', logging.INFO, formatter, os.path.join(log_save_dir, 'IoU.txt'))
        logger_miou = get_logger('miou', logging.INFO, formatter, os.path.join(log_save_dir, 'mIoU.txt'))
        logger_fwiou = get_logger('fwiou', logging.INFO, formatter, os.path.join(log_save_dir, 'FWIoU.txt'))

        loss_list, pa_list, cpa_list, mpa_list, iou_list, miou_list, fwiou_list = [], [], [], [], [], [], []

        logger_log.info(
            f'checkpoint_save_dir {checkpoint_save_dir}'
            f', pretrained_backbone {opts.pretrained_backbone}'
            f', crop_size {opts.crop_size}'
            f', num_epochs {opts.num_epochs}'
            f', batch_size {opts.batch_size}'
            # f', aux_loss {opts.aux_loss}'
            f', weight_decay {opts.weight_decay}'
            f', base_lr {opts.base_lr}'
            f', poly_lr_power {opts.poly_lr_power}'
        )


    # dataset
    if not opts.val_only:
        train_transform = et.ExtCompose([
            et.ExtRandomCrop(opts.crop_size, pad_if_needed=True, fill=255),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = VOCSegDataset(opts.voc_root, True, train_transform)
        train_iter = torch.utils.data.DataLoader(train_dataset, opts.batch_size, shuffle=True, num_workers=4)
        logger_log.info(f'Train Set {len(train_dataset)} Samples, Batches {len(train_iter)}.')

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = VOCSegDataset(opts.voc_root, False, val_transform)
    val_iter = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True, num_workers=2)
    logger_log.info(f'Validate Set {len(val_dataset)} Samples.')


    # train and validate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3plus.__dict__[opts.model](
        weights_backbone=opts.weights_backbone,
        pretrained_backbone=opts.pretrained_backbone,
        num_classes=opts.num_classes,
        # aux_loss=opts.aux_loss,
        output_stride=opts.output_stride,
    )

    val_metric = metrics.SegmentationMetric(opts.num_classes)
    if opts.val_only:
        logger_log.info('------------------------------ Only Validate ------------------------------')

        if opts.checkpoint_path and os.path.isfile(opts.checkpoint_path):
            logger_log.info(f'Loading Checkpoint From {opts.checkpoint_path}.')
            checkpoint = torch.load(opts.checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state'])
            logger_log.info(f'Checkpoint Score {checkpoint["score"]}.')

        model = nn.DataParallel(model)
        model.to(device)
        validate(model, val_iter, device, val_metric)

        results = val_metric.get_results()
        logger_log.info('Validate Metric:')
        logger_log.info(f'\tPA: \t{results["PA"]}')
        logger_log.info(f'\tcPA: \t{results["cPA"]}')
        logger_log.info(f'\tmPA: \t{results["mPA"]}')
        logger_log.info(f'\tIoU: \t{results["IoU"]}')
        logger_log.info(f'\tmIoU: \t{results["mIoU"]}')
        logger_log.info(f'\tFWIoU: \t{results["FWIoU"]}')

        scio.savemat(os.path.join(log_save_dir, 'confusion_matrix.mat'), {'data': val_metric.confusion_matrix})
        logger_log.info('------------------------------ Only Validate End ------------------------------')
        return

    logger_log.info('------------------------------ Training ------------------------------')
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(
        params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.base_lr if opts.pretrained_backbone else opts.base_lr},
            {'params': model.classifier.parameters(), 'lr': opts.base_lr},
        ], 
        lr=opts.base_lr,
        momentum=0.9,
        weight_decay=opts.weight_decay,
    )
    scheduler = poly_lr_scheduler.PolyLR(optimizer, opts.num_epochs * len(train_iter), opts.poly_lr_power)
    model = nn.DataParallel(model)
    model.to(device)
    
    def save_checkpoint(path, score):
        torch.save(
            {
                "model_state": model.module.state_dict(),
                "score": score
            },
            path,
        )
    
    epoch, best_score = 0, 0

    while True:
        epoch += 1
        sum_loss, interval_loss = 0, 0
        
        for i, (features, labels) in enumerate(train_iter):
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            model.train()
            optimizer.zero_grad()
            outputs = model(features)['out']
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            sum_loss += loss.item()
            interval_loss += loss.item()

            if (i + 1) % 300 == 0:
                interval_loss /= 300
                print(f'Batch {i + 1}/{len(train_iter)}, Loss {interval_loss:.6f}')
                interval_loss = 0

        validate(model, val_iter, device, val_metric)
        results = val_metric.get_results()

        logger_log.info(
            f'Epoch {epoch}'
            f', Loss {sum_loss / len(train_iter)}'
            f', Backbone LR {scheduler.get_lr()[0]}'
            f', Classifier LR {scheduler.get_lr()[1]}'
        )
        
        logger_log.info('Validate Metric:')
        logger_log.info(f'\tPA: \t{results["PA"]}')
        logger_log.info(f'\tcPA: \t{results["cPA"]}')
        logger_log.info(f'\tmPA: \t{results["mPA"]}')
        logger_log.info(f'\tIoU: \t{results["IoU"]}')
        logger_log.info(f'\tmIoU: \t{results["mIoU"]}')
        logger_log.info(f'\tFWIoU: \t{results["FWIoU"]}')

        logger_loss.info(f'{sum_loss / len(train_iter)}')
        logger_pa.info(results["PA"])
        logger_cpa.info(results["cPA"])
        logger_mpa.info(results["mPA"])
        logger_iou.info(results["IoU"])
        logger_miou.info(results["mIoU"])
        logger_fwiou.info(results["FWIoU"])

        loss_list.append(sum_loss / len(train_iter))
        pa_list.append(results["PA"])
        cpa_list.append(results["cPA"])
        mpa_list.append(results["mPA"])
        iou_list.append(results["IoU"])
        miou_list.append(results["mIoU"])
        fwiou_list.append(results["FWIoU"])

        save_checkpoint(os.path.join(checkpoint_save_dir, 'latest.pth'), results["mIoU"])
        print(f'Updata Latest Checkpoint To: {os.path.join(checkpoint_save_dir, "latest.pth")}.')
        
        if best_score < results["mIoU"]:
            best_score = results["mIoU"]
            save_checkpoint(os.path.join(checkpoint_save_dir, 'best.pth'), results["mIoU"])
            print(f'Updata Best Checkpoint To: {os.path.join(checkpoint_save_dir, "best.pth")} (Score {best_score}).')

        if epoch >= opts.num_epochs:
            scio.savemat(
                os.path.join(log_save_dir, 'confusion_matrix.mat'), 
                {
                    'confusion_matrix': val_metric.confusion_matrix,
                    'loss_list': loss_list,
                    'pa_list': pa_list,
                    'cpa_list': cpa_list,
                    'mpa_list': mpa_list,
                    'iou_list': iou_list,
                    'miou_list': miou_list,
                    'fwiou_list': fwiou_list
                }
            )
            break

    logger_log.info('------------------------------ Training End ------------------------------')


if __name__ == '__main__':
    main()
