import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

pcap_file = 'D:/PycharmProjects/Resnet_TC_NEW/data_1_14/service/train-pcap.npy'
label_file = 'D:/PycharmProjects/Resnet_TC_NEW/data_1_14/service/train-labels.npy'
label2index = {'Chat': 0, 'Email': 1, 'FT': 2, 'P2P': 3, 'Streaming': 4, 'VoIP': 5, 'VPN_Chat': 6, 'VPN_Email': 7,
               'VPN_FT': 8, 'VPN_P2P': 9, 'VPN_Streaming': 10, 'VPN_VoIP': 11}
index2label = {0: 'Chat', 1: 'Email', 2: 'FT', 3: 'P2P', 4: 'Streaming', 5: 'VoIP', 6: 'VPN_Chat', 7: 'VPN_Email',
               8: 'VPN_FT', 9: 'VPN_P2P', 10: 'VPN_Streaming', 11: 'VPN_VoIP'}
pcap_data = np.load(pcap_file)
label_data = np.load(label_file)
print('pcap  {}; label {}.'.format(pcap_data.shape, label_data.shape))

def save_pcap_image(pcap_name, pcap_index_list, pcap_data):
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.05, wspace=0.001)
    axs = axs.ravel()

    for i, j in enumerate(pcap_index_list):
        axs[i].axis("off")
        image_data = pcap_data[j]
        image_data = image_data.reshape(28, 28)
        im = Image.fromarray(image_data)
        axs[i].imshow(im, cmap='gray', vmin=0, vmax=255)

    plt.savefig('{}.jpg'.format(pcap_name))


def save_mean_pcap_image(pcap_name, pcap_index_list, pcap_data):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.axis("off")

    mean_image_data = np.mean(pcap_data[pcap_index_list], axis=0)
    mean_image_data = mean_image_data.reshape(28, 28)

    im = Image.fromarray(mean_image_data)
    axs.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.savefig('{}_mean.jpg'.format(pcap_name))

for label_name, label_index in label2index.items():
    pcap_index_list = np.where(label_data == label_index)[0]
    save_mean_pcap_image(label_name, pcap_index_list, pcap_data)