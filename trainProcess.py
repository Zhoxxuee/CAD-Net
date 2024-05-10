from TrafficFlowClassification.utils.helper import AverageMeter, accuracy
from TrafficFlowClassification.TrafficLog.setLog import logger

def train_process(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (pcap, target) in enumerate(train_loader):
        pcap = pcap.to(device)
        target = target.to(device)
        output = model(pcap)
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), pcap.size(0))
        top1.update(prec1[0].item(), pcap.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1))
    return losses.avg, top1.avg
