import numpy as np
import h5py


class Hammer(object):

    def __init__(self, outfile_name, num_step, num_walk, num_param, log_prob_func, log_prob_args):
        
	self.outfile_name = outfile_name
        self.num_step = num_step
        self.num_walk = num_walk
        self.num_param = num_param
        self.log_prob_func = log_prob_func
        self.log_prob_args = log_prob_args
        
        self.num_sample = self.num_step * self.num_walker
        self.samples = np.zeros((self.num_step, self.num_walk, self.num_param))       
