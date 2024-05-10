import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from TrafficFlowClassification.TrafficLog.setLog import logger
from TrafficFlowClassification.utils.calc_class_weight import calculate_weights_1
from TrafficFlowClassification.utils.setConfig import setup_config
from torch.autograd import Variable
import numpy as np
from TrafficFlowClassification.models.cnn1d import cnn1d
from TrafficFlowClassification.models.se_resnet18_2d import se_resnet182D
from TrafficFlowClassification.train.trainProcess import train_process
from TrafficFlowClassification.train.validateProcess import validate_process
from TrafficFlowClassification.data.dataLoader import data_loader
from TrafficFlowClassification.data.tensordata import get_tensor_data
from TrafficFlowClassification.utils.helper import adjust_learning_rate, save_checkpoint
from TrafficFlowClassification.utils.evaluate_tools import display_model_performance_metrics
from TrafficFlowClassification.utils.evaluate_tools import display_confusion_matrix_plot

def train_pipeline():
    cfg = setup_config()
    logger.info(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(' GPU ?, {}'.format(device))
    model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name)
    model = DCSE_resnet182d(model_path,
                  pretrained=cfg.test.pretrained,
                  num_classes=cfg.train.num_class,
                  image_width=cfg.train.IMAGE_WIDTH).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    device = torch.device("cuda:0")
    model.to(device)

    labels_count = [713, 771, 8238, 311, 831, 11596, 201, 114, 348, 204, 380, 558]
    weights = calculate_weights_1(labels_count)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
    # criterion = MultiFocalLoss(num_class=12, gamma=2.0, reduction='mean', alpha=weights_final)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    # optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr, momentum=0.9)
    train_loader = data_loader(pcap_file=cfg.train.train_pcap,
                               label_file=cfg.train.train_label,
                               trimed_file_len=cfg.train.TRIMED_FILE_LEN,
                               batch_size=cfg.train.BATCH_SIZE)
    test_loader = data_loader(pcap_file=cfg.train.test_pcap,
                              label_file=cfg.train.test_label,
                              trimed_file_len=cfg.train.TRIMED_FILE_LEN,
                              batch_size=cfg.train.BATCH_SIZE)
    if cfg.test.evaluate:
        validate_process(test_loader, model, criterion, device, 20)
        torch.cuda.empty_cache()
        if cfg.train.num_class == 12:
            index2label = {j: i for i, j in
                           cfg.test.label2index.items()}
            list_max = 3200
        if cfg.train.num_class == 20:
            index2label = {j: i for i, j in
                           cfg.test.label2index_USTC_APP.items()}
            list_max = 10100
        label_list = [index2label.get(i) for i in range(cfg.train.num_class)]
        pcap_data, label_data = get_tensor_data(pcap_file=cfg.train.val_pcap,
                                                label_file=cfg.train.val_label,
                                                trimed_file_len=cfg.train.TRIMED_FILE_LEN)
        start_index = 0
        y_pred = None
        for i in list(range(100, list_max, 100)):
            y_pred_batch = model(pcap_data[start_index:i].to(device))
            start_index = i
            if y_pred == None:
                y_pred = y_pred_batch.cpu().detach()
            else:
                y_pred = torch.cat((y_pred, y_pred_batch.cpu().detach()), dim=0)
                print(y_pred.shape)
        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)
        Y_data_label = [index2label.get(i.tolist()) for i in label_data]
        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]
        display_model_performance_metrics(true_labels=Y_data_label,
                                          predicted_labels=pred_label,
                                          classes=label_list)

        display_confusion_matrix_plot(true_labels=Y_data_label,
                                      predicted_labels=pred_label,
                                      classes=label_list)
        return
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_prec1 = 0
    for epoch in tqdm(range(cfg.train.epochs), desc='Training', unit="epoch"):
        if cfg.train.lr_adjust:
            adjust_learning_rate(optimizer, epoch, cfg.train.lr)

        train_loss, train_accuracy = train_process(train_loader, model, criterion, optimizer, epoch, device, 300)
        prec1 = train_accuracy
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_accuracy = validate_process(test_loader, model, criterion, device, 60)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

    logger.info('Finished')
    # folder_path = '/home/zab/PycharmProjects/Resnet_TC_12_4/checkpoint/811/ac_loss_list'
    #
    # train_accuracies_list = np.array(train_accuracies)
    # val_accuracies_list = np.array(val_accuracies)
    # train_losses_list = np.array(train_losses)
    # val_losses_list = np.array(val_losses)
    #
    # np.save(f'{folder_path}/{cfg.train.model_name}_train_acc.npy', train_accuracies_list)
    # np.save(f'{folder_path}/{cfg.train.model_name}_val_acc.npy', val_accuracies_list)
    # np.save(f'{folder_path}/{cfg.train.model_name}_train_loss.npy', train_losses_list)
    # np.save(f'{folder_path}/{cfg.train.model_name}_val_loss.npy', val_losses_list)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.ylim((0, 100))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.ylim((0, 4))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_pipeline()
