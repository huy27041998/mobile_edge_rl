import numpy as np

class EdgeEnergyEnv:
    def __init__(self, method):
        #lambda: total work time t
        #ht: network congestion
        #bt: battery
        #et: environment state
        super().__init__()
        self.k = 0  
        self.d_sta = 0
        self.timeslot = 0.25  # hours, ~15min
        self.batery_capacity = 2000  # Wh
        self.server_service_rate = 20  # units/sec
        self.lambda_high = 100  # units/second
        self.lambda_low = 10
        self.b_high = self.batery_capacity / self.timeslot  # W
        self.b_low = 0
        self.h_high = 0.06*5  # s/unit
        self.h_low = 0.02*5
        self.back_up_cost_coef = 0.15
        self.normalized_unit_depreciation_cost = 0.01
        self.max_number_of_server = 15
        self.g = []
        self.t = 0
        self.method = method
        self.cost = [0 for i in range(96)]
        self.alpha = 0.1
        self.gamma = 1
        # power model
        self.d_sta = 300
        self.coef_dyn = 0.5
        self.server_power_consumption = 150
        self.time_steps_per_episode = 96
        self.episode = 0
        self.state = [0, 0, 0, 0] #lambdat, bt, ht, et

    def init(self):
        self.state[0] = self.get_lambdat()
        self.update_bt()
        self.state[2] = self.get_ht()
    def get_a_t(self, t):
        #power consumption for autosacling
        #action handling
        return

    def get_ut(self):
        return 

    def get_mt(self):
        return 1
    
    def update_bt(self):
        self.state[1] = max(self.state[1] + self.get_gt(), self.b_high)

    def get_gt(self):
        return self.g[self.get_t()]

    def get_ht(self):
        return np.random.uniform(self.h_low, self.h_high)

    def get_lambdat(self):
        return np.random.uniform(self.lambda_low, self.lambda_high)
    
    def get_t(self):
        return int(self.t * 4)

    def get_c_local(self, at):
        # delay tu user den local
        ut = self.state[0] - at
        mt = self.get_mt()
        if ut == 0 and mt == 0:
            return 0
        return ut / (mt * self.server_service_rate - ut)

    def get_c_off(self, at):
        # khoi luong xu ly off_server x ping den off_server
        ut = self.state[0] - at
        return (self.state[0] - ut) * self.state[2]

    def get_cwi_t(self, t):
        return 0
    
    def get_cdelay_t(self, at):
        return self.get_c_local(at) + self.get_c_off(at)

    def get_d_dyn(self):
        return self.coef_dyn * self.state[0]

    def get_dop(self):
        return self.d_sta + self.get_d_dyn()

    def getdcom(self):
        return max([self.state[1] - self.get_dop(), 0])
    
    def get_dt(self):
        return self.getdcom() + self.get_dop()
        
    def calculate_action(self):
        ats = [i for i in range(0, int(self.state[1]))]
        return self.find_max_action(ats)

    def find_max_action(self, ats):
        if self.method == 'qlearning':
            cost = 0
            best_action = 0
            for at in ats:
                new_cost = self.get_cdelay_t(at) + at
                if (new_cost < cost):
                    cost = new_cost
                    best_action = at
        return new_cost, best_action
    
    def update_state(self, new_cost, at):
        self.state[0] = self.get_lambdat()
        self.state[1] -= at
        self.state[2] = self.get_ht()
        self.cost[self.get_t()] += self.alpha * ((1 + self.gamma) * new_cost - self.cost[self.get_t()])
        self.t += 0.25
        if self.t == 24:
            self.t = 0