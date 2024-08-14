import level_inter as li

def get_datasets(opts):
    train_transform = li.transforms.Compose([
        li.transforms.ToTensor(),
    ])
    val_transform = li.transforms.Compose([
        li.transforms.ToTensor(),
        ])
    train_path = li.os.path.join(opts.data_root,'whu/train')
    val_path = li.os.path.join(opts.data_root,'whu/val')

    train_dst = li.whu_dataset(train_path,transform=train_transform)
    val_dst = li.whu_dataset(val_path, transform=val_transform)
    return train_dst,val_dst



def validate(opts,net,loader,device,bce_fn,dice_fn,best_score,epoch,start_time,total_time):
    val_loss = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (image, label) in enumerate(loader):
        net.eval()
        image, label = image.to(device), label.to(device)
        output = net(image)
        loss = li.Combined_Bce_Dice_Loss(output, label, bce_fn, dice_fn)
        val_loss += loss.cpu().item()
        output = li.torch.where(output >= 0.5, 1, 0)
        re = li.tp_fn(label, output)
        tp += re[0]
        tn += re[1]
        fp += re[2]
        fn += re[3]

    val_loss /= len(loader.dataset)
    # 计算一个epoch的accuray、recall、precision、IOU
    total_recall = li.Recall(tp, fn)
    total_precision = li.Precision(tp, fp)
    total_iou = li.IoU(tp, fp, fn)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    end_time = li.time.time()
    total_time += (end_time - start_time)
    print('val set: Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}, IoU:{:.4f}, Loss:{:.4f}'.format(total_precision,
                                                                                                total_recall, total_f1,
                                                                                                total_iou, val_loss/opts.val_batch_size))
    # print(total_iou)
    print('本轮耗时:{}'.format(li.datetime.timedelta(seconds=total_time)))
    res = 'epoch:{} val set: Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}, IoU:{:.4f}, Loss:{:.4f}\n'.format(epoch+1,total_precision,
                                                                                                total_recall, total_f1,
                                                                                                total_iou, val_loss/opts.val_batch_size)

    f = open('./result/whu', 'a', encoding='utf-8')
    f.write(res)
    f.close()

    if best_score < total_iou:
        best_score = total_iou
        li.torch.save(net.state_dict(), './result/whu/{}.mdl'.format(epoch+1))
    elif total_iou > 0.90:
        li.torch.save(net.state_dict(), './result/whu/_{}.mdl'.format(epoch+1))

    return best_score

# 构建网络
def main():
    opts = li.get_argparser().parse_args()
    net = li.CHINet()
    if li.os.path.exists(opts.model_path):
        net.load_state_dict(li.torch.load(opts.model_path))
        print(True)

    device = 'cuda'
    net.to(device)


    train_dst,val_dst = get_datasets(opts)
    train_loader = li.DataLoader(train_dst, batch_size=opts.batch_size,
                                shuffle=True,drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = li.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          ('whu', len(train_dst), len(val_dst)))

    bce_fn = li.nn.BCELoss()
    dice_fn = li.SoftDiceLoss()
    bce_fn.to(device)
    dice_fn.to(device)

    best_score = 0.0
    total_time = 0
    optimizer = li.torch.optim.Adam(
        net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay,
    )

    scheduler = li.optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=[20, 50, 80, 100,  150, 190],
                                                  gamma=0.1)
    for epoch in range(opts.total_itrs):
        start_time = li.time.time()


        # 训练阶段
        net.train()
        train_loss = 0.0
        count = 0
        for i,(img,lab) in enumerate(train_loader):

            # 训练数据的个数
            img,lab = img.to(device),lab.to(device)
            out = net(img)
            loss = li.Combined_Bce_Dice_Loss(out, lab, bce_fn, dice_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            if (i+1) % 200 == 0 and (i+1) != len(train_loader):
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(epoch+1,opts.total_itrs,(i+1)*opts.batch_size,len(train_loader.dataset),
                                                                                    100. * (i + 1) / len(train_loader),
                                                                                    train_loss / (500 * opts.batch_size)))
                train_loss = 0.0
                count += 1
            elif i+1 == len(train_loader):
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(epoch + 1, opts.total_itrs,(i + 1) * opts.batch_size,len(train_loader.dataset),
                                                                                    100. * (i + 1) / len(train_loader),
                                                                                    train_loss / (len(train_loader.dataset)-count*500*opts.batch_size)))

        scheduler.step()

        # 验证阶段
        best_score= validate(opts,net,val_loader,device,bce_fn,dice_fn,best_score,epoch,start_time,total_time)

if __name__ == '__main__':
    main()