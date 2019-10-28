import numpy as np

def Thresh2Obj(latency_list,thresh):
	[sent,drop],_ = np.histogram(latency_list,bins=[0,thresh,10])
	return 1.0*sent/thresh

def FindThresh(latency_list):
	max_obj = None
	T = None
	for thresh in range(1,100):
		obj = Thresh2Obj(latency_list,thresh*0.01)
		if not max_obj or obj>max_obj:
			max_obj = obj
			T = thresh*0.01
	return T