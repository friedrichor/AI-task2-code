import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import params
from my_dataset import MyDataSet
from utils import split_train_val, read_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


def main(args):
    print(args)
    device = params.device
    print(f"using {device} device.")

    tb_writer = SummaryWriter()

    # train_images_path, train_images_label, val_images_path, val_images_label = split_train_val(args.data_path)
    train_images_path, train_images_label = read_data(args.train_path, 'train')
    val_images_path, val_images_label = read_data(args.val_path, 'val')

    img_size = params.img_size
    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomResizedCrop((img_size, img_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            # transforms.Resize((int(img_size * 1.143), int(img_size * 1.143))),
            # transforms.CenterCrop(img_size),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)


    model = params.model.to(device)
    if args.weights != '':
        print('have pre weights')
        weights = os.path.join(params.path_weights, args.weights)
        model.load_state_dict(torch.load(weights, map_location=device))
    model_path = os.path.join(params.path_weights, args.model_name + '.pth')  # 模型保存路径

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=10)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), model_path)
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    parser.add_argument('--img-size', type=int, default=params.img_size)
    parser.add_argument('--loss-fun', type=str, default=params.loss_function._get_name())
    # 模型保存名称
    parser.add_argument('--model-name', type=str, default='')
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default=params.path_data)
    parser.add_argument('--train-path', type=str, default=params.path_train)
    parser.add_argument('--val-path', type=str, default=params.path_test)

    # 预训练权重路径
    parser.add_argument('--weights', type=str,
                        default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()
    main(opt)
