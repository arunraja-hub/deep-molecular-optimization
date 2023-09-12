import argparse

import configuration.opts as opts
from trainer.transformer_trainer import TransformerTrainer
# from trainer.seq2seq_trainer import Seq2SeqTrainer


import optuna
from optuna.trial import TrialState

import torch
import torch.multiprocessing as mp

import os

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))
        sockname = str(s.getsockname()[1])
        print("free_port-" ,sockname)
        return sockname

if __name__ == "__main__":
    print('main')
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.train_opts(parser)
    opt = parser.parse_args()

    if opt.model_choice == 'transformer':
        trainer = TransformerTrainer(opt)
    # elif opt.model_choice == 'seq2seq':
    #     trainer = Seq2SeqTrainer(opt)

    
    # torch.cuda.set_device(opt.local_rank)
    world_size = torch.cuda.device_count()
    # int(os.environ["WORLD_SIZE"])
    # 3   
    rank = int(os.environ['SLURM_PROCID'])
    # opt.local_rank
    # int(os.environ['SLURM_PROCID'])
    # 0 

    # print( 'args in train',int(os.environ["WORLD_SIZE"]),int(os.environ['SLURM_PROCID']), int(os.environ['LOCAL_RANK']))


    free_port = '12347'
    # find_free_port()

    dist_url = ''
    #  f'tcp://10.140.21.19:{free_port}'

    print('spawn')
    mp.spawn(
        trainer.train(opt,rank,world_size,free_port,dist_url),
        args=(),
        nprocs=world_size
    )

    
    