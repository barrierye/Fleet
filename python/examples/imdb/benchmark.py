# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import time
import requests
from imdb_reader import IMDBDataset
from paddle_serving_client import Client
from paddle_serving_client.utils import MultiThreadRunner
from paddle_serving_client.utils import benchmark_args

args = benchmark_args()

def single_func(idx, resource):
    imdb_dataset = IMDBDataset()
    imdb_dataset.load_resource(args.vocab)
    filelist_fn = args.filelist
    filelist = []
    start = time.time()
    with open(filelist_fn) as fin:
        for line in fin:
            filelist.append(line.strip())
    filelist = filelist[idx::args.thread]
    if args.request == "rpc":
        client = Client()
        client.load_client_config(args.model)
        client.connect([args.endpoint])
        for fn in filelist:
            fin = open(fn)
            for line in fin:
                word_ids, label = imdb_dataset.get_words_and_label(line)
                fetch_map = client.predict(feed={"words": word_ids},
                                           fetch=["prediction"])
    elif args.request == "http":
        for fn in filelist:
            fin = open(fn)
            for line in fin:
                word_ids, label = imdb_dataset.get_words_and_label(line)
                r = requests.post("http://{}/imdb/prediction".format(args.endpoint),
                                  data={"words": word_ids, "fetch": ["prediction"]})
    end = time.time()
    return [[end - start]]

multi_thread_runner = MultiThreadRunner()
result = multi_thread_runner.run(single_func, args.thread, {})
print(result)