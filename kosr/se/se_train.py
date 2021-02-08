import torch
import warnings
import yaml
import argparse
import sys

sys.path.append('../../')

warnings.filterwarnings('ignore')

from kosr.se.models import build_model
from kosr.utils.loss import build_criterion
from kosr.utils.optimizer import build_optimizer
#from kosr.data.dataset import get_dataloader

def build_conf(conf_path='config/ksponspeech_se_base.yaml'):
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    
    return conf

def main(args):
    conf = build_conf(args.conf)
    
    batch_size = conf['train']['batch_size']
    
    #train_dataloader = get_dataloader(conf['dataset']['train'], batch_size=batch_size, mode='train', conf=conf)
    #valid_dataloader = get_dataloader(conf['dataset']['valid'], batch_size=batch_size, conf=conf)
    #test_dataloader = get_dataloader(conf['dataset']['test'], batch_size=batch_size, conf=conf)
    
    model = build_model(conf)
    criterion = build_criterion(conf)
    optimizer = build_optimizer(model.parameters(), **conf['optimizer'])
    
    saved_epoch = load(args, model, optimizer)
    
    train_and_eval(conf['train']['epochs'], model, optimizer, criterion, train_dataloader, valid_dataloader, epoch_save=True, saved_epoch=saved_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End Speech Recognition Training')
    parser.add_argument('--conf', default='config/ksponspeech_se_base.yaml', type=str, help="configuration path for se training")
    parser.add_argument('--continue_from', default='', type=str, help="continue to train from saved se model")
    args = parser.parse_args()
    main(args)