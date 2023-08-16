import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 180/np.pi
MPS_TO_KMH = 12/5
KMH_TO_MPS = 5/12

# Road params
ROAD_WIDTH = 3.7*2 # meters
ROAD_LENGTH = 200 # meters

# Vehicle dynamics - Bicycle Model
class Vehicle():

    def __init__(self, x_init = np.zeros(8), uc_init= np.zeros(2), solve = True):

        # Initialize state vector and control variable
        self.name = "Ego"
        self.x = x_init
        self.uc = uc_init
        # self.t_span = t_span
        # self.t_step = t_step

        self.x_simulated = None
        self.t_sim = None
        self.simtype = "zero"
        self.solve = solve

        # Vehicle Dimensions:
        self.mass = 2271 # Kg
        self.vlength = 4.5
        self.vwidth = 1.8
        self.bbox_offset = 0.1
        self.diag_len = np.sqrt((self.vlength/2 + self.bbox_offset)**2 + (self.vwidth/2 + self.bbox_offset)**2) 
        self.bbox_coords = None

        self.plotscalex = 5
        self.plotscaley = 0.25

    def set_state(self, x):
        self.x = x

    def get_control_input(self, t, **kwargs):

        if self.simtype == "control":
            self.uc[0] = kwargs["d"]
            self.uc[1] = kwargs["F"]

        elif self.simtype == "zero":
            self.uc[0] = 5 * DEG_TO_RAD
            self.uc[1] = 0

        elif self.simtype == "zigzag":

            start = 1
            if(t < start):
                self.uc[0] = 0
                self.uc[1] = 0
            else:
                if int(t / 3) % 2 == 0: 
                    self.uc[0] = 5 * DEG_TO_RAD
                    self.uc[1] = 0
                else:
                    self.uc[0] = -5 * DEG_TO_RAD
                    self.uc[1] = 0

    def bicycle_model(self, t, x):
    
        # State variables
        u = x[0]
        v = x[1]
        r = x[2]
        theta = x[3]
        X = x[4]
        Y = x[5]
        d = x[6]
        F = x[7]

        # if u<=40*KMH_TO_MPS: Fc = 4*self.mass
        # else :Fc = 0

        # Control variables
        # self.get_control_input(t, d = 0, F = Fc)

        delta = self.uc[0]
        FxT = self.uc[1]

        # Steering angle saturation
        delta_d = (delta - d)
        deltad_max = 5 * DEG_TO_RAD # maximum of 40 deg/sec turn rate

        # steering rate saturation
        if np.abs(delta_d) > deltad_max:
            delta_d = np.sign(delta_d) * deltad_max

        # Steering angle saturation

        m = 2271 # Kg
        Iz = 4600 #Kgm^2
        lf = 1.421 #m
        lr = 1.434 #m
        Cf = 132000 #N
        Cr = 136000 #N
        
        # Max Total longitudinal Force
        FxT_max = 24800 #N max acceleration = 11m/s^2, max deceleration = -15*m N, comfortable acc = 4*m N, comfortable deceleration = -2*m.
        # Max Total lateral force on front tyre 
        Fyf_max = 10400 #N
        # Max Total lateral force on rear tyre 
        Fyr_max = 10600 #N
        
        # Total Force
        FxT = min(FxT, FxT_max)
        d_F = (FxT - F)

        # Total lateral force on front tyre
        Fyf = min(Cf*(delta - (v + lf*r)/u), Fyf_max)
        # Total lateral force on rear tyre
        Fyr = min(-Cr*(v - lr*r)/u, Fyr_max)

        #print(FxT, Fyf, Fyr)

        u_dot = FxT/m + v*r
        v_dot = (Fyf + Fyr)/m - u*r
        r_dot = (lf*Fyf - lr*Fyr)/Iz
        theta_dot = r
        X_dot = u*np.cos(theta) - v*np.sin(theta)
        Y_dot = v*np.cos(theta) + u*np.sin(theta)

        x_dot = np.array([u_dot,v_dot,r_dot,theta_dot,X_dot,Y_dot,delta_d,d_F])

        return x_dot
    
    def set_bbox(self):

        if(self.x_simulated is not None):

            sol = self.x_simulated
            theta = sol[3][-1]
            x = sol[4][-1]
            y = sol[5][-1]

            tl = [-self.vlength/2-self.bbox_offset, +self.vwidth/2+self.bbox_offset]
            tr = [+self.vlength/2+self.bbox_offset, +self.vwidth/2+self.bbox_offset]
            dr = [+self.vlength/2+self.bbox_offset, -self.vwidth/2-self.bbox_offset]
            dl = [-self.vlength/2-self.bbox_offset, -self.vwidth/2-self.bbox_offset]
            bbox = np.array([tl,tr,dr,dl])
            
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            bbox = bbox.dot(R.T)
            bbox[:,0] += x
            bbox[:,1] += y
            bbox_r = np.append(bbox[:,0], bbox[0,0])
            bboy_r = np.append(bbox[:,1], bbox[0,1])
            self.bbox_coords = np.array([bbox_r,bboy_r])

        else:
            print("Empty history, execute simulation first.")


    # def solve_ode(self):
    #     t_eval = np.linspace(self.t_span[0],self.t_span[1], num = int(self.t_span[1]/self.t_step))
    #     self.x_simulated = solve_ivp(self.bicycle_model, t_span = self.t_span, y0 = self.x, t_eval = t_eval, method='RK45', dense_output=False)

    def LineLineCollision(self, l1, l2):
        
        # Line 1
        x1 = l1[0,0]
        y1 = l1[0,1]
        x2 = l1[1,0]
        y2 = l1[1,1]

        # Line 2
        x3 = l2[0,0]
        y3 = l2[0,1]
        x4 = l2[1,0]
        y4 = l2[1,1]
        
        denom = ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))

        if denom == 0: return False

        uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom

        if (uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1): return True

        return False

    def CheckCollision(self, obs):

        # Lane collision
        for i in range(4):
            l1 = self.bbox_coords[:, i : i + 2].T.copy()
            l2 = l1.copy()
            l2[:,1] = ROAD_WIDTH # Left Lane
            l3 = l1.copy()
            l3[:,1] = 0 # Right Lane
            if self.LineLineCollision(l1,l2) or self.LineLineCollision(l1,l3): return True

        # Vehicle collision
        xe, ye = self.x_simulated[4][-1], self.x_simulated[5][-1]
        for ob in obs:
            xo, yo = self.x_simulated[4][-1], self.x_simulated[5][-1]
            if np.sqrt((xe-xo)**2 + (ye - yo)**2) < 2 * self.diag_len:

                for i in range(4):
                    l1 = self.bbox_coords[:, i : i + 2].T
                    for i in range(4):
                        l2 = ob.bbox_coords[:, i : i + 2].T
                        if self.LineLineCollision(l1,l2): return True

        return False

class Road():

    def __init__(self, width = ROAD_WIDTH, length = ROAD_LENGTH):
        # Initialize state vector and control variable
        self.width = width
        self.hw = width / 2 
        self.length = length
        self.l_max = -np.Inf
        self.l_min = np.Inf
        self.plot_vel = False
        self.draw_bobx = True

    def plot_road_solution(self, vehicles):

        fig, ax = plt.subplots()
        vehicle_color = iter(["r","b","g","y"])
        for vehicle in vehicles:

            if(vehicle.x_simulated is not None):

                time_sol = vehicle.t_sim
                x_sol = vehicle.x_simulated

                u = x_sol[0]
                v = x_sol[1]
                r = x_sol[2]
                theta = x_sol[3]
                X = x_sol[4]
                Y = x_sol[5]
                d = x_sol[6]
                F = x_sol[7]

                self.l_max = max(np.max(X), self.l_max)
                self.l_min = min(np.min(X), self.l_min)
                self.length = max(self.length, self.l_max - self.l_min)

                ax.plot(X,Y)
                ax.set_xlabel("x-axis [m]")
                ax.set_ylabel("y-axis [m]")

                # Plot car
                x_car = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0,  0])*vehicle.vlength
                y_car = np.array([  -1,    1,    1,   0,   -1,   -1,    0,   0,    1, 1, -1])*vehicle.vwidth/2
                
                # Draw car body, first and last
                m_indx = [-1]
                v_color = next(vehicle_color)
                for m_i in m_indx:
                    x_new_car = X[m_i] + x_car * np.cos(theta[m_i]) - y_car * np.sin(theta[m_i])
                    y_new_car = Y[m_i] + x_car * np.sin(theta[m_i]) + y_car * np.cos(theta[m_i])
                    ax.plot(x_new_car, y_new_car, v_color)

                # Draw Bbox
                if self.draw_bobx:
                    x = vehicle.bbox_coords[0]
                    y = vehicle.bbox_coords[1]
                    ax.plot(x, y, color='black', linestyle='--')

            else:
                print("Empty history, execute simulation first.")

        self.draw_road_section(ax)

        # Plot velocities
        if self.plot_vel:
            for vehicle in vehicles:
                if(vehicle.x_simulated is not None):

                    time_sol = vehicle.t_sim
                    x_sol = vehicle.x_simulated

                    u = x_sol[0]
                    v = x_sol[1]
                    r = x_sol[2]
                    theta = x_sol[3]
                    X = x_sol[4]
                    Y = x_sol[5]
                    d = x_sol[6]
                    F = x_sol[7]

                    if vehicle.solve:

                        plt.plot(time_sol, u*MPS_TO_KMH)
                        plt.xlabel("Time [s]")
                        plt.ylabel("Speed [km/h]")
                        plt.show()

                        plt.plot(time_sol, v*MPS_TO_KMH)
                        plt.xlabel("Time [s]")
                        plt.ylabel("lateral Speed [km/h]")
                        plt.show()

                        plt.plot(time_sol, d*RAD_TO_DEG)
                        plt.xlabel("Time [s]")
                        plt.ylabel("Steering angle [deg]")
                        plt.show()

                        plt.plot(time_sol, F)
                        plt.xlabel("Time [s]")
                        plt.ylabel("FxT")
                        plt.show()

    def draw_road_section(self, ax):

        # Define road boundaries
        left_boundary = self.width
        right_boundary = 0

        # Draw road boundaries
        ax.plot([0, self.length], [left_boundary, left_boundary], color='gray', linewidth=5)
        ax.plot([0, self.length], [right_boundary, right_boundary], color='gray', linewidth=5)

        # Draw lane markings
        center_boundary = self.hw
        ax.plot([0, self.length], [center_boundary, center_boundary], color='black', linestyle='--', linewidth=5)
        ax.axis("equal")

        plt.show()

def SolverFun(T, X, vehicles):
    x_out = []
    v_count = 0
    for vehicle in vehicles:
        if vehicle.solve:
            x = X[8*v_count : 8*(v_count+1)]
            x_out.append(vehicle.bicycle_model(T,x))
            v_count += 1
    x_out = np.concatenate(x_out)
    return x_out


def ODESolver(vehicles, t_span = (0,0.1)):

    #ODE Solver
    X0 = []
    # Initial state vector
    for vehicle in vehicles:
        if vehicle.solve:
            X0.append(vehicle.x)
    X0 = np.concatenate(X0)
    #t_eval = np.linspace(t_span[0],t_span[1],int((t_span[1]-t_span[0])/0.1))
    t_eval = t_span
    x_sim_net = solve_ivp(SolverFun, t_span = t_span, y0 = X0, t_eval = t_eval, method='RK45', dense_output = True, args=[vehicles])
    v_count = 0

    for vehicle in vehicles:

        if vehicle.solve:
            # Moving Vehicle
            vehicle.x_simulated = x_sim_net.y[8*v_count : 8*(v_count+1)]
            vehicle.t_sim = x_sim_net.t
            v_count += 1
        else:
            # Stationary vehicle
            vehicle.x_simulated = np.array([vehicle.x]).T
        vehicle.set_bbox()

if __name__ == "__main__":
    
    # Make ego vehicle
    t_span = (0,15) # 20 sec solution
    t_step = 0.1 # Time step
    x_e = np.zeros(8)
    x_e[0] = 0.01 * KMH_TO_MPS # u = 40 km/hr ~ 25 Miles/Hr
    x_e[4] = 0
    x_e[5] = 3*ROAD_WIDTH/4 
    x_e[3] = (0/180)*np.pi # Heading
    ego = Vehicle(x_init = x_e)
    ego.simtype = "zero"

    # Make obstacle vehicle - stationary
    x_o1 = np.zeros(8)
    x_o1[0] = 0 * KMH_TO_MPS # u = 40 km/hr
    x_o1[4] = 100
    x_o1[5] = 3*ROAD_WIDTH/4
    x_o1[3] = 0 # Heading
    obs1 = Vehicle(x_init = x_o1, solve = False)
    obs1.name = "Obs1"

    # Make obstacle vehicle - stationary
    x_o2 = np.zeros(8)
    x_o2[0] = 30 * KMH_TO_MPS # u = 40 km/hr
    x_o2[4] = 300
    x_o2[5] = ROAD_WIDTH/4
    x_o2[3] = np.pi # Heading
    obs2 = Vehicle(x_init = x_o2)
    obs2.name = "Obs2"

    # Make Road
    road = Road()

    vehicles = [ego,obs1]
    obs_set = [obs1]

    ODESolver(vehicles, t_span = t_span)
    # road.plot_vel = True
    # road.draw_bobx = False

    #print(ego.CheckCollision(obs_set))
    road.plot_road_solution(vehicles)
    
    #from apf import APF
    #pot = APF(length = road.length, width = road.width)
    #pot.plot_potential(ego, obs_set)