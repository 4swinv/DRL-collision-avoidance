import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

MPS_TO_KMH = 18/5
KMH_TO_MPS = 5/18

ROAD_WIDTH = 3.7*2
ROAD_LENGTH = 200 # meters

class APF():

    def __init__(self, width = ROAD_WIDTH, length = ROAD_LENGTH):
        
        self.width = width
        self.center = width/2
        self.length = length

        # Road potential params
        self.yc1 = width/4
        self.yc2 = 3*width/4
        self.pm = 0.5

        # Obstacle potential params
        self.delta_x = 50 * 0.1
        self.delta_y = 2.5 * 0.1
        self.epsilon = 10
        self.Uobs = 0

        # Grid params for plotting
        self.xx = None
        self.yy = None
        self.U = None
    
    # Two-way lane road potential
    def Road_potential_2way_opp(self, y):

        Amp = np.array(y > self.yc2, dtype=int) * (1 - self.pm) + self.pm
        w = 2*np.pi*y / (self.width/2)
        Uroad = np.cos(w) - (np.array(y < self.center, dtype=int)) * ((np.cos(w) - 1) - (1-self.pm)*(self.center-y))
        Uroad = 0.5 * (Amp*(Uroad + 1)) #+ (1-self.pm)*(np.array(y < self.yc1, dtype=int)) * (np.cos(w) + 1))
    
        return Uroad
    
    def Obstacle_potential(self, ego, obs, plot = False):
        
        sole = ego.x_simulated

        U_obs = 0

        if plot:
            ue = 40*KMH_TO_MPS
            ve = 0
            Xe = self.xx
            Ye = self.yy
        else:
            ue = sole[0][-1]
            ve = sole[1][-1]
            Xe = sole[4][-1]
            Ye = sole[5][-1]

        vele = np.sqrt( ue**2 + ve**2 ) 

        for ob in obs:

            sol = ob.x_simulated 
            uo = sol[0][-1]
            vo = sol[1][-1]
            Xo = sol[4][-1]
            Yo = sol[5][-1]

            velo = np.sqrt( uo**2 + vo**2)
            v_rel = vele - velo

            U_obs = np.exp( - (1 /  (v_rel + self.epsilon)) * ( (Xe-Xo)**2 / self.delta_x**2 + (Ye-Yo)**2 / self.delta_y**2 ) ) 

        return U_obs

    def plot_potential(self, ego, obs):

        x = np.linspace(0, self.length, int(self.length/0.5))
        y = np.linspace(0, self.width, int(self.length/0.1))

        self.xx, self.yy = np.meshgrid(x,y)
        self.U = np.zeros(self.xx.shape)

        self.U += self.Road_potential_2way_opp(self.yy)

        self.U += self.Obstacle_potential(ego, obs, plot = True)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        surf = ax.plot_surface(self.xx, self.yy, self.U, cmap = cm.jet)
        ax.set_box_aspect((np.ptp(self.xx)/10, np.ptp(self.yy), np.ptp(self.U)))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('U Label')
        plt.show()



        
