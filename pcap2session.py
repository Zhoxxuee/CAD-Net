import os
import subprocess
from TrafficFlowClassification.TrafficLog.setLog import logger

def pcap_to_session(pcap_folder, splitcap_path):
    splitcap_path = os.path.normpath(splitcap_path)
    for (root, _, files) in os.walk(pcap_folder):
        for Ufile in files:
            pcap_file_path = os.path.join(root, Ufile)
            pcap_name = Ufile.split('.')[0]
            pcap_suffix = Ufile.split('.')[1]
            try:
                assert pcap_suffix == 'pcap'
            except:
                logger.warning('pcapng')
                assert pcap_suffix == 'pcap'
            os.makedirs(os.path.join(root, pcap_name), exist_ok=True)
            prog = subprocess.Popen([splitcap_path,
                                     "-p", "100000",
                                     "-b", "100000",
                                     "-r", pcap_file_path,
                                     "-o", os.path.join(root, pcap_name)],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            _, _ = prog.communicate()
            os.remove(pcap_file_path)
    logger.info('pcap to session.')
    logger.info('============\n')
