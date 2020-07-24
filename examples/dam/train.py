#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Deep Attention Matching Network
"""
import sys
import os
import six
import numpy as np
import time
import multiprocessing
import paddle
import paddle.fluid as fluid
import reader as reader
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
import evaluation as eva
from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

from model_check import check_cuda
import utils
import config
import model

def main():
    args = config.parse_args()
    config.print_arguments(args)
    
    # load data
    with open(args.data_path, 'rb') as f:
        if six.PY2:
            train_data, val_data, test_data = pickle.load(f)
        else:
            train_data, val_data, test_data = pickle.load(
                    f, encoding="bytes")

    if args.do_train:
        if args.distributed:
            init_dist_env()

        # create executor and place
        # (multi machine mode is different from single machine mode)
        place = create_place(args.distributed)
        exe = create_executor(place)

        # create train network
        train_prog, start_prog = fluid.Program(), fluid.Program()
        with fluid.program_guard(train_prog, start_prog):
            feed, fetch, optimizer = model.build_train_net(args)

        if args.distributed:
            # distributed  optimizer
            optimizer = distributed_optimize(optimizer)    
        optimizer.minimize(fetch[0], start_prog)

        # create train dataloader
        # (multi machine mode is different from single machine mode)
        loader = create_train_dataloader(args, train_data, feed, place, args.distributed)
        if args.distributed:
            # be sure to do the following assignment before executing
            # train_prog to specify the program encapsulated by the 
            # distributed policy
            train_prog = fleet.main_program

        # do train
        print('train')
        train(args, train_prog, start_prog, exe, feed, fetch, loader)

    if args.do_test:
        assert args.distributed == False 

        place = create_place(False)
        exe = create_executor(place)

        model_path = "saved_model/model"

        # create test network
        test_prog, start_prog = fluid.Program(), fluid.Program()
        with fluid.program_guard(test_prog, start_prog):
            feed, fetch = model.build_test_net(args)
            fluid.io.load_persistables(
                    executor=exe,
                    dirname=model_path,
                    main_program=fluid.default_main_program())

        # create test dataloader
        loader = create_test_dataloader(args, test_data, feed, place, False)
        # test on one card
        test(args, test_prog, exe, feed, fetch, loader)

def init_dist_env():
    """
    init distributed env by fleet
    """
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

def create_place(is_distributed):
    """
    decide which device to use based on distributed env
    """
    place_idx = int(os.environ['FLAGS_selected_gpus']) if is_distributed else 0
    return fluid.CUDAPlace(place_idx)

def distributed_optimize(optimizer):
    strategy = DistributedStrategy()
    strategy.fuse_all_reduce_ops = True
    strategy.nccl_comm_num = 2 
    strategy.fuse_elewise_add_act_ops=True
    strategy.fuse_bn_act_ops = True
    return fleet.distributed_optimizer(optimizer, strategy=strategy)

def create_executor(place):
    exe = fluid.Executor(place)
    return exe

def create_train_dataloader(args, data, feed, place, is_distributed):
    return create_dataloader(args, data, feed, place, is_distributed, False)

def create_test_dataloader(args, data, feed, place, is_distributed):
    return create_dataloader(args, data, feed, place, is_distributed, True)

def create_dataloader(args, data, feed, place, is_distributed, is_test):
    data_conf = {
        "batch_size": 1, # args.batch_size,
        "max_turn_num": args.max_turn_num,
        "max_turn_len": args.max_turn_len,
        "_EOS_": args._EOS_,
    }
    batch_num = len(data[six.b('y')]) // args.batch_size

    def batch_generator(data, batch_num):
        def generator():
            if not is_test:
                shuffle_data = reader.unison_shuffle(data, seed=None)
            else:
                shuffle_data = data
            data_batches = reader.build_batches(shuffle_data, data_conf)
            for index in six.moves.xrange(batch_num):
                yield reader.make_one_batch_input(data_batches, index)
        return generator

    return utils.create_dataloader(
            batch_generator(data, batch_num),
            feed, place, batch_size=args.batch_size,
            is_test=is_test, is_distributed=is_distributed)

def train(args, train_prog, start_prog, exe, feed, fetch, loader):
    exe.run(start_prog)
    for epoch in range(args.num_scan_data):
        for idx, sample in enumerate(loader()):
            ret = exe.run(train_prog, feed=sample, fetch_list=fetch)
            if idx % 1 == 0:
                print('[TRAIN] epoch=%d step=%d loss=%f' % (epoch, idx, ret[0][0]))

    save_path = "saved_model/model"
    if args.distributed and fleet.worker_index() == 0:
        fleet.save_persistables(executor=exe, dirname=save_path)
        print("model saved in {}".format(save_path))

def test(args, test_prog, exe, feed, fetch, loader):
    filename = "score"
    score_file = open(filename, 'w')

    fetch_list = fetch + ["label"]
    for idx, sample in enumerate(loader()):
        ret = exe.run(test_prog, feed=sample, fetch_list=fetch_list)
        scores = np.array(ret[0])
        label = np.array(ret[1])
        for i in six.moves.xrange(len(scores)):
            score_file.write("{}\t{}\n".format(scores[i][0], int(label[i][0])))
    score_file.close()

    result = eva.evaluate_ubuntu(filename)
    print("[TEST] result: ")
    for metric in result:
        print("[TEST]   {}: {}".format(metric, result[metric]))
    return filename

if __name__ == '__main__':
    main()
