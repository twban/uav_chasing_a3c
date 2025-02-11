import numpy as np
class Args() : 
    def __init__(self,cnn, initial_dist_target, initial_space_drones, doppler, corr_coeff, n_rssi_samples, succ_thres_dist=2, tx_pow = 23):

        # 1. Environment
        self.n_drones = 6 # 3D, x+, x-, y+, y-, z+, z-
        self.initial_dist_target = initial_dist_target
        self.space_drones = initial_space_drones
        self.doppler = doppler
        self.corr_coeff = corr_coeff
        
            # Channel modeling
        self.path_exp = 2.6 # Path loss exponent
        self.D0 = 1 # Reference distance
        self.L0 = 30 # Path loss measured at a reference distance D0
        self.tx_pow = tx_pow # transmit power of target at dBm
        self.succ_thres_dist = succ_thres_dist # Success if distance btw target - tracker center < succ_thres_dist

        # 2. Deep Learning Neural Model
        self.cnn = cnn
        self.n_rssi_samples = n_rssi_samples
            # LSTM Layer
        self.n_hiddens = 128
        self.n_layers = 3        
            # Convolutional and Linear Layer
        self.n_features = self.n_hiddens


        # 3. Agent
            # Action
        self.n_actions = 5 # should be odd
        self.step_s = 1
        self.actions_s = list( np.arange(1, self.n_actions+1)*self.step_s ) #step=1->drone_space [1, 2, 3, 4, 5], step=2->[2, 4, 6, 8, 10]         
        self.step_xyz = 2
        self.actions_xyz = list(  (np.arange(0, self.n_actions) - int((self.n_actions-1)/2) )  * self.step_xyz )  # [-2*step, -step, 0, step, 2*step]

        self.max_train_ep = 200
        self.test_interval = 100
        self.n_train_processes = 8 # 8 is good
        self.update_interval = 5 # 5 is good
        
        self.learning_rate = 1e-5 #1e-5 is good
        self.entropy_coeff = 0.01 # 0.01 good
        self.gamma = 0.99
        
        self.fail_cnt = 500
        self.print_every = 5
        self.stop_succ_cnt = 100
        self.device = 'cuda'


    def __call__(self):
        pass