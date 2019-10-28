import numpy as np
import os
import collections
import sys
sys.path.insert(0, '/home/bo/research/xprotocol')
from util.ThreshFinder import FindThresh

data_dir = "../data/"
latency_file = data_dir + "latency.dat"
trace_file = data_dir + "trace.dat"

# read latency
label_raw_train,label_raw_test = collections.defaultdict(list),collections.defaultdict(list)
label_train,label_test = collections.defaultdict(list),collections.defaultdict(list)
all_latency = []
with open(latency_file) as fp:
    line = fp.readline()
    latency_log = []
    desc = None
    key = ""
    while line:
        if line[0]=='S':
            # store previous result, if any
            if latency_log:
                if desc[-1]=='1':
                    label_raw_test[key] += latency_log
                else:
                    label_raw_train[key] += latency_log
            # initialize all variables
            latency_log = []
            desc = line.strip().split('\t')
            key = str(desc[1:-1])
        else:
            # add latency to log
            latency = float(line)
            latency_log.append(latency)
            all_latency.append(latency)

        line = fp.readline()
    if latency_log:
        if desc[-1]=='1':
            label_raw_test[key] += latency_log
        else:
            label_raw_train[key] += latency_log

# find threshold
thresh = FindThresh(all_latency)
print("Setting the threshold to:",thresh)


# label all data based on threshold
for desc in label_raw_train:
    label_train[desc] = [0 if t<=thresh else 1 for t in label_raw_train[desc]]

for desc in label_raw_test:
    label_test[desc] = [0 if t<=thresh else 1 for t in label_raw_test[desc]]


# sizes of 1000 files
all_sizes = []
for i in range(1000):
    filename = "/var/www/Bigbunny/%04d.h264"%(i+1)
    statinfo=os.stat(filename)
    all_sizes.append(statinfo.st_size)


ping_buf_size = 10
frame_buf_size = 5
# data with ping latency, frame latency and frame size
data_train,data_test = collections.defaultdict(list),collections.defaultdict(list)
# data with frame latency and frame size
data_train2,data_test2 = collections.defaultdict(list),collections.defaultdict(list)
# data with frame latency and frame size but got via normal acks
# to be filled here
# read traces
with open(trace_file) as fp:
    line = fp.readline()
    ping_buf = []
    frame_buf = []
    size_buf = []
    data_buf = []
    desc = None
    key = ""
    while line:
        if line[0]=='S':
            # normalize collected data
            if len(ping_buf)>0:
                mean_ping,std_ping = np.mean(ping_buf),np.std(ping_buf)
                mean_latency,std_latency = np.mean(frame_buf),np.std(frame_buf)
                mean_size,std_size = np.mean(size_buf),np.std(size_buf)
                tmp,tmp2 = [],[]
                for x,s in zip(data_buf,all_sizes):
                    ping,latency,size = x[:ping_buf_size],x[ping_buf_size:ping_buf_size+frame_buf_size],x[ping_buf_size+frame_buf_size:ping_buf_size+frame_buf_size*2]
                    ping = [(p-mean_ping)/std_ping if p>0 else 0 for p in ping]
                    latency = [(l-mean_latency)/std_latency if l>0 else 0 for l in latency]
                    size = [(s-mean_size)/std_size if s>0 else 0 for s in size]
                    tmp.append(ping+latency+size+[s])
                    tmp2.append(latency+size+[s])
                if desc[-1]=='1':
                    data_test[key] += tmp
                    data_test2[key] += tmp2
                else:
                    data_train[key] += tmp
                    data_train2[key] += tmp2

            ping_buf = []
            frame_buf = []
            size_buf = []
            data_buf = []
            desc = line.strip().split('\t')
            key = str(desc[1:-1])
            print ("Processing:",desc)
        else:
            if len(line)>=6 and line[:6]=="[XACK]":
                pass
            elif len(line)>=6 and line[:6]=="[SEND]":
                x = []
                if len(ping_buf)>=ping_buf_size:
                    x += ping_buf[-ping_buf_size:]
                else:
                    x += [0]*(ping_buf_size-len(ping_buf)) + ping_buf
                if len(frame_buf)>=frame_buf_size:
                    x += frame_buf[-frame_buf_size:]
                    x += size_buf[-frame_buf_size:]
                else:
                    x += [0]*(frame_buf_size-len(frame_buf)) + frame_buf
                    x += [0]*(frame_buf_size-len(frame_buf)) + size_buf
                data_buf.append(x)
            elif len(line)>=7 and line[:7]=="[PROBE]":
                num_probes = int(line[7:])
                for pidx in range(num_probes):
                    probe_line = fp.readline()
                    probe_t = float(probe_line.split('\t')[0])
                    ping_buf.append(probe_t)
            elif len(line)>=7 and line[:7]=="[FRAME]":
                num_frames = int(line[7:])
                for fidx in range(num_frames):
                    frame_line = fp.readline()
                    frame_t,frame_size = frame_line.split('\t')[:2]
                    frame_buf.append(float(frame_t))
                    size_buf.append(float(frame_size))
            else:
                print ("Error",line)
                exit(1)
        line = fp.readline()

    # normalize collected data
    mean_ping,std_ping = np.mean(ping_buf),np.std(ping_buf)
    mean_latency,std_latency = np.mean(frame_buf),np.std(frame_buf)
    mean_size,std_size = np.mean(size_buf),np.std(size_buf)
    tmp,tmp2 = [],[]
    for x,s in zip(data_buf,all_sizes):
        ping,latency,size = x[:ping_buf_size],x[ping_buf_size:ping_buf_size+frame_buf_size],x[ping_buf_size+frame_buf_size:ping_buf_size+frame_buf_size*2]
        ping = [(p-mean_ping)/std_ping if p>0 else 0 for p in ping]
        latency = [(l-mean_latency)/std_latency if l>0 else 0 for l in latency]
        size = [(s-mean_size)/std_size if s>0 else 0 for s in size]
        tmp.append(ping+latency+size+[s])
        tmp2.append(latency+size+[s])
    if desc[-1]=='1':
        data_test[key] += tmp
        data_test2[key] += tmp2
    else:
        data_train[key] += tmp
        data_train2[key] += tmp2

print(len(data_train),len(data_test))

# save data
np.save('../data/preprocessed/proposed/data_train.npy',data_train)
np.save('../data/preprocessed/proposed/label_train.npy',label_train)
np.save('../data/preprocessed/proposed/data_test.npy',data_test)
np.save('../data/preprocessed/proposed/label_test.npy',label_test)

np.save('../data/preprocessed/withoutprobe/data_train.npy',data_train2)
np.save('../data/preprocessed/withoutprobe/label_train.npy',label_train)
np.save('../data/preprocessed/withoutprobe/data_test.npy',data_test2)
np.save('../data/preprocessed/withoutprobe/label_test.npy',label_test)

# prepare data for input from only frame latency
# still need a trace with server sends and server receiving normal acks indicating the receiving time for comparison
