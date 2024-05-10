import torch
import numpy as np
from TrafficFlowClassification.TrafficLog.setLog import logger
def data_loader(pcap_file, label_file, trimed_file_len, batch_size=128, workers=4, pin_memory=False):
    pcap_data = np.load(pcap_file)
    label_data = np.load(label_file)
    pcap_data = torch.from_numpy(pcap_data.reshape(-1, 1, trimed_file_len)).float()
    label_data = torch.from_numpy(label_data).long()
    logger.info(
        'pcap , {}; label  {}'.format(pcap_data.shape, label_data.shape))

    res_dataset = torch.utils.data.TensorDataset(pcap_data, label_data)
    res_dataloader = torch.utils.data.DataLoader(
        dataset=res_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=4
    )
    print(res_dataloader)
    return res_dataloader