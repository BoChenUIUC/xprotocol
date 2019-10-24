import numpy as np

data_dir = "../data/"
latency_file = data_dir + "latency.dat"
trace_file = data_dir + "trace.dat"

# read latency
all_latency_log = {}
with open(latency_file) as fp:
    line = fp.readline()
    latency_log = []
    desc = ""
    while line:
        if line[0]=='S':
            # store previous result, if any
            if latency_log:
                all_latency_log[desc] = latency_log
            # initialize all variables
            latency_log = []
            desc = line
        else:
            # add latency to log
            latency = float(line)
            latency_log.append(latency)

        line = fp.readline()
    if latency_log:
        all_latency_log[desc] = latency_log

# hist = np.histogram(raw_latency_log,bins=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,10])


def latency_to_label(latency):
    if latency<0.05:return 0
    if latency<0.1:return 1
    if latency<0.15:return 2
    if latency<0.2:return 3
    if latency<0.25:return 4
    if latency<0.3:return 5
    if latency<0.35:return 6
    if latency<0.4:return 7
    if latency<0.45:return 8
    return 9

ping_buf_size = 10
frame_buf_size = 5
# data with ping latency, frame latency and frame size
data_train,data_test = [],[]
label_train,label_test = [],[]
# data with frame latency and frame size
data_train2,data_test2 = [],[]
# data with frame latency and frame size but got via normal acks
# to be filled here
# read traces
with open(trace_file) as fp:
    line = fp.readline()
    ping_buf = []
    frame_buf = []
    size_buf = []
    data_buf = []
    desc = ""
    while line:
        if line[0]=='S':
            # normalize collected data
            if len(ping_buf)>0:
                mean_ping,std_ping = np.mean(ping_buf),np.std(ping_buf)
                mean_latency,std_latency = np.mean(frame_buf),np.std(frame_buf)
                mean_size,std_size = np.mean(size_buf),np.std(size_buf)
                assert(std_ping>0 and std_latency>0 and std_size>0)
                tmp,tmp2 = [],[]
                for x in data_buf:
                    ping,latency,size = x[:ping_buf_size],x[ping_buf_size:ping_buf_size+frame_buf_size],x[ping_buf_size+frame_buf_size:ping_buf_size+frame_buf_size*2]
                    ping = [(p-mean_ping)/std_ping for p in ping]
                    latency = [(l-mean_latency)/std_latency for l in latency]
                    size = [(s-mean_size)/std_size for s in size]
                    tmp.append(ping+latency+size)
                    tmp2.append(latency+size)
                if desc.strip()[-1]=='1':
                    data_test += tmp
                    data_test2 += tmp2
                    label_test += [latency_to_label(t) for t in all_latency_log[desc]]
                else:
                    data_train += tmp
                    data_train2 += tmp2
                    label_train += [latency_to_label(t) for t in all_latency_log[desc]]
                assert(len(data_test)==len(label_test))
                assert(len(data_train)==len(label_train))

            ping_buf = []
            frame_buf = []
            size_buf = []
            data_buf = []
            desc = line
            print "Processing:",desc
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
                print "Error",line
                exit(1)
        line = fp.readline()

    # normalize collected data
    mean_ping,std_ping = np.mean(ping_buf),np.std(ping_buf)
    mean_latency,std_latency = np.mean(frame_buf),np.std(frame_buf)
    mean_size,std_size = np.mean(size_buf),np.std(size_buf)
    assert(std_ping>0 and std_latency>0 and std_size>0)
    tmp,tmp2 = [],[]
    for x in data_buf:
        ping,latency,size = x[:ping_buf_size],x[ping_buf_size:ping_buf_size+frame_buf_size],x[ping_buf_size+frame_buf_size:ping_buf_size+frame_buf_size*2]
        ping = [(p-mean_ping)/std_ping for p in ping]
        latency = [(l-mean_latency)/std_latency for l in latency]
        size = [(s-mean_size)/std_size for s in size]
        tmp.append(ping+latency+size)
        tmp2.append(latency+size)
    if desc[-1]=='1':
        data_test += tmp
        data_test2 += tmp2
    else:
        data_train += tmp
        data_train2 += tmp2
    if desc[-1]=='1':
        label_test += [latency_to_label(t) for t in all_latency_log[desc]]
    else:
        label_train += [latency_to_label(t) for t in all_latency_log[desc]]


# save data
np.save('../data/preprocessed/proposed/data_train.npy',data_train)
np.save('../data/preprocessed/proposed/label_train.npy',label_train)
np.save('../data/preprocessed/proposed/data_test.npy',data_test)
np.save('../data/preprocessed/proposed/label_train.npy',label_test)

np.save('../data/preprocessed/withoutprobe/data_train.npy',data_train2)
np.save('../data/preprocessed/withoutprobe/label_train.npy',label_train)
np.save('../data/preprocessed/withoutprobe/data_test.npy',data_test2)
np.save('../data/preprocessed/withoutprobe/label_train.npy',label_test)

# prepare data for input from only frame latency
# still need a trace with server sends and server receiving normal acks indicating the receiving time for comparison
