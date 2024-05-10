import torch
import time
from TrafficFlowClassification.utils.helper import AverageMeter, accuracy
from TrafficFlowClassification.TrafficLog.setLog import logger

def validate_process(val_loader, model, criterion, device, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    for i, (pcap, target) in enumerate(val_loader):

        pcap = pcap.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(pcap)
            loss = criterion(output, target)
            prec1 = accuracy(output.data, target)
            losses.update(loss.item(), pcap.size(0))
            top1.update(prec1[0].item(), pcap.size(0))
            if (i + 1) % print_freq == 0:
                logger.info(
                    'Test: [{0}/{1}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), loss=losses, top1=top1))
    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg
