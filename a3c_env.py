import numpy as np
import math
import scipy.io as io

# https://ko.wikipedia.org/wiki/%EA%B5%AC%EB%A9%B4%EC%A2%8C%ED%91%9C%EA%B3%84


#****** All distances and Movemenet are based on from origin of tracker and Target
class DroneTrackingENV() : 
    def __init__(self, args, mode='TRAIN'):
        self.args = args
        if mode == 'TRAIN':
            file = f"ch_sample10_doppler0{int(self.args.doppler*10)}_b0{int(self.args.corr_coeff*10)}.mat"
        if mode == 'TEST':
            file = f"ch_sample10_doppler0{int(self.args.doppler*10)}_b0{int(self.args.corr_coeff*10)}_test.mat"
        
        mat = io.loadmat("./channel/"+file)
        self.fading = mat['powers_norm_db']

        self.n_rssi_samples = self.args.n_rssi_samples
        self.succ_thres_dist = self.args.succ_thres_dist
        self.actions_s = self.args.actions_s
        self.actions_xyz = self.args.actions_xyz

        # Channel model
        self.doppler = self.args.doppler
        self.corr_coeff = self.args.corr_coeff
        self.path_exp = self.args.path_exp # Path loss exponent
        self.D0 = self.args.D0 # Reference distance
        self.L0 = self.args.L0 # Path loss measured at a reference distance D0
        self.tx_pow = self.args.tx_pow # transmit power of target at dBm
        
        _ = self.reset()
        
    def reset(self) :
        self.dist_target = self.args.initial_dist_target
        
        # set target coordinates randomly with given dist_target
        self.place_target()

        self.space_drones = self.args.space_drones
        self.tracker_loc = [0, 0, 0] # Center of tracker
        # set drones_initial locations
        self.update_drones_locs()

        # index of starting measuring rssi 
        self.ch_idx = np.random.randint(low=0, high=10000)
        state = self.build_state()
        return state
    
    def cal_distance(self, loc1, loc2) : 
        return math.dist(loc1, loc2)

    def place_target(self) : 
        # r: distance from origin to target
        phi = np.random.randint(0, 360) * np.pi / 180
        theta = np.random.randint(0, 180) * np.pi / 180
        target_x = self.args.initial_dist_target * math.sin(theta) * math.cos(phi)
        target_y = self.args.initial_dist_target * math.sin(theta) * math.sin(phi)
        target_z = self.args.initial_dist_target * math.cos(theta)
        
        self.target_loc = [target_x, target_y, target_z]

    def update_drones_locs(self):
        #tracker_loc: location of origin (center) of racking system 
        x, y, z = self.tracker_loc 
        # x+, x-, y+, y-, z+, z-
        self.drones_locs = [\
                        [x + self.space_drones, y, z], \
                        [x - self.space_drones, y, z], \
                        [x, y + self.space_drones, z], \
                        [x, y - self.space_drones, z], \
                        [x, y, z + self.space_drones], \
                        [x, y, z - self.space_drones]
                    ]

    def build_state(self) : 
        dists_drones_target = [self.cal_distance(self.target_loc, drone_loc) for drone_loc in self.drones_locs]
        self.path_losses = np.array( [self.L0 + 10 * self.path_exp * np.log10((dist+1e-8)/(self.D0+1e-8)) for dist in dists_drones_target] )
        rssis = self.tx_pow - self.path_losses + self.fading[self.ch_idx:self.ch_idx + self.n_rssi_samples]
        self.ch_idx = self.ch_idx + self.n_rssi_samples
        if self.fading.shape[0] - self.ch_idx < 5*self.args.n_rssi_samples:
            self.ch_idx = np.random.randint(low=0, high=10000)
        state = rssis
        return state
    
    def move(self, actions) :
        idx_s, idx_x, idx_y, idx_z = actions
        s, x, y, z = self.actions_s[idx_s], self.actions_xyz[idx_x], self.actions_xyz[idx_y], self.actions_xyz[idx_z]
        
        # 1. move the origin of tracker
        self.tracker_loc_old = self.tracker_loc
        self.tracker_loc = [self.tracker_loc[0]+x, self.tracker_loc[1]+y, self.tracker_loc[2]+z] 
        
        # 2. adjust the space between drones
        self.space_drones = s
        
        # 3. update locations of 6 drones
        self.update_drones_locs()
        
        next_state = self.build_state()

        # 4. get reward
        new_dist_target = self.cal_distance(self.target_loc, self.tracker_loc) 
        reward = [self.dist_target - new_dist_target] # reward of S
        self.xyz_dist_old = np.abs( np.array(self.tracker_loc_old) - np.array(self.target_loc) )
        self.xyz_dist_new = np.abs( np.array(self.tracker_loc) - np.array(self.target_loc) )
        reward_xyz = self.xyz_dist_old - self.xyz_dist_new
        for r in reward_xyz:
            reward.append(r) # S, X, Y, Z

        # 5. Done or not
        if (np.array([self.cal_distance(self.target_loc,drone_loc) for drone_loc in self.drones_locs])<self.succ_thres_dist).any():# or new_dist_target < self.succ_thres_dist:
            done = True
        else:
            done = False

        if done: reward = [10]*4
        
        # 6. Update distance btw tracker and target
        self.dist_target = new_dist_target
        
        return next_state, reward, done

