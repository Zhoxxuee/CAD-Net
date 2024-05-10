import torch
import numpy as np
from TrafficFlowClassification.TrafficLog.setLog import logger

def get_tensor_data(pcap_file, label_file, trimed_file_len):
    pcap_data = np.load(pcap_file)
    label_data = np.load(label_file)
    pcap_data = torch.from_numpy(pcap_data.reshape(-1, 1, trimed_file_len)).float()
    label_data = torch.from_numpy(label_data).long()
    
    return pcap_data, label_data