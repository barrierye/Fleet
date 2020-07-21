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
from util import mkdir
import evaluation as eva
import config

try:
    import cPickle as pickle  #python 2
except ImportError as e:
    import pickle  #python 3

from model_check import check_cuda
from net import Net

def batch_generator(train_batches, batch_num):
    def generator():
        for index in six.moves.xrange(batch_num):
            yield make_one_batch_input(train_batches, index)
    return generator

def create_dataloader(feed_list, capacity, drop_last, train_batches, batch_num):
    feed_list = self.turns_data + self.turns_mask \
            + [self.response] + [self.response_mask] + [self.label]
    loader = fluid.io.DataLoader.from_generator(
            feed_list=feed_list, 
            capacity=capacity,
            drop_last=drop_last)
    loader.set_batch_generator(
            batch_generator(train_batches, batch_num),
            fluid.cpu_places())
    return loader

def train(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # data data_config
    data_conf = {
        "batch_size": args.batch_size,
        "max_turn_num": args.max_turn_num,
        "max_turn_len": args.max_turn_len,
        "_EOS_": args._EOS_,
    }

    dam = Net(args.max_turn_num, args.max_turn_len, args.vocab_size,
              args.emb_size, args.stack_num, args.channel1_num,
              args.channel2_num)

    # dataloader params
    capacity = 10
    drop_last = True # drop_last cannot be False when train
    filelist = ["data_small.pkl"]
    feed_list = dam.get_feed_list()

    # create net
    train_program = fluid.Program()
    train_startup = fluid.Program()
    with fluid.program_guard(train_program, train_startup):
        with fluid.unique_name.guard():
            loss, logits = dam.create_network()
            loss.persistable = True
            logits.persistable = True
            # gradient clipping
            fluid.clip.set_gradient_clip(clip=fluid.clip.GradientClipByValue(
                max=1.0, min=-1.0))

            optimizer = fluid.optimizer.Adam(
                learning_rate=fluid.layers.exponential_decay(
                    learning_rate=args.learning_rate,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True))
            optimizer.minimize(loss)

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
        dev_count = fluid.core.get_cuda_device_count()
    else:
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

    print("device count %d" % dev_count)
    print("theoretical memory usage: ")
    print(fluid.contrib.memory_usage(
        program=train_program, batch_size=args.batch_size))

    exe = fluid.Executor(place)
    exe.run(train_startup)

    train_exe = fluid.ParallelExecutor(
        use_cuda=args.use_cuda, loss_name=loss.name, main_program=train_program)

    if args.word_emb_init is not None:
        print("start loading word embedding init ...")
        if six.PY2:
            word_emb = np.array(pickle.load(open(args.word_emb_init,
                                                 'rb'))).astype('float32')
        else:
            word_emb = np.array(
                pickle.load(
                    open(args.word_emb_init, 'rb'), encoding="bytes")).astype(
                        'float32')
        dam.set_word_embedding(word_emb, place)
        print("finish init word embedding  ...")
    
    print("start loading data ...")
    with open(args.data_path, 'rb') as f:
        if six.PY2:
            train_data, val_data, test_data = pickle.load(f)
        else:
            train_data, val_data, test_data = pickle.load(f, encoding="bytes")
    print("finish loading data ...")
    batch_num = len(train_data[six.b('y')]) // data_conf["batch_size"]

    print("begin model training ...")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    
    def train_one_epoch(step, dataloader):
        """
        Train one epoch
        """
        ave_cost = 0.0
        for data in dataloader():
            cost = train_exe.run(feed=data, fetch_list=[loss.name])

            ave_cost += np.array(cost[0]).mean()
            step = step + 1

        return step, np.array(cost[0]).mean()

    # train over different epoches
    global_step, train_time = 0, 0.0
    for epoch in six.moves.xrange(args.num_scan_data):
        shuffle_train = reader.unison_shuffle(
            train_data, seed=None)
        train_batches = reader.build_batches(shuffle_train, data_conf)

        # create dataloader
        dataloader = create_dataloader(feed_list, capacity,
                drop_last, train_batches, batch_num)

        begin_time = time.time()
        global_step, last_cost = train_one_epoch(global_step, dataloader)

        pass_time_cost = time.time() - begin_time
        train_time += pass_time_cost
        print("Pass {0}, pass_time_cost {1}"
              .format(epoch, "%2.2f sec" % pass_time_cost))

if __name__ == '__main__':
    args = config.parse_args()
    config.print_arguments(args)

    check_cuda(args.use_cuda)

    if args.do_train:
        train(args)

    if args.do_test:
        test(args)
