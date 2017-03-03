from bicycle_env import BalanceTask
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.learners.valuebased.linearfa import LSPI
from pybrain.rl.experiments import EpisodicExperiment
import numpy as np
from matplotlib import pyplot as plt

class LSPI_task(BalanceTask):
    def __init__(self,*args,**kwargs):
        self.max_tilt = np.pi / 12.0
        BalanceTask.__init__(self, *args,**kwargs)
    def get_features(self):
        (th,th_d,om,omg_d,omg_dd,_,_,_,_,psi) = self.env.sensors
        psig = self.env.psig
        if psig >= 0:
            psig_new = np.pi - psig
        else:
            psig_new = -np.pi - psig
        X = self.env.sensors
        x_2 = np.power(X[2],2)
        x_0 = np.power(X[0],2)
        res = [1,X[2],X[3],x_2,np.power(X[3],2),X[2]*X[3],X[0],X[1],x_0,np.power(X[1],2),
              X[0]*X[1],X[2]*X[0],X[2]*x_0,x_2*X[0],psig,np.power(psig,2),psig*X[0],psig_new,np.power(psig_new,2),psig_new*X[0]]
        return res




        #pass
    def getObservation(self):
        return self.get_features()

    def isFinished(self):
        target = np.array([30,50])
        (_, _, _, _, _,xf, yf, _, _, _) = self.env.sensors
        dist_to_goal = np.linalg.norm(target-np.array([xf,yf]))
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return True
        if dist_to_goal < 1e-2:
            return True
        return False
    def getReward(self):
        target = np.array([30,50])
        (_, _, _, _, _,xf, yf, _, _, _) = self.env.sensors
        dist_to_goal = np.linalg.norm(target-np.array([xf,yf]))
        delta_tilt = self.env.getTilt()**2 - self.env.last_omega**2
        last_xf = self.env.last_xf
        last_yf = self.env.last_yf
        dist_to_goal_last = np.linalg.norm(target-np.array([last_xf,last_yf]))
        delta_dist = dist_to_goal - dist_to_goal_last
        return -delta_tilt - delta_dist * 0.01

task = LSPI_task()
learner = LSPI(9,20)
task.rewardDiscount = 0.8
learner.rewardDiscount = 0.8

agent = LinearFA_Agent(learner)
agent.epsilonGreedy = True
exp = EpisodicExperiment(task, agent)
learner.learningRateDecay = 3000
max_agent = LinearFA_Agent(learner)
max_agent.learnerning = False
max_agent.greedy  = True

task.env.saveWheelContactTrajectories(True)
plt.ion()
plt.figure(figsize=(8, 4))

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

def update_wheel_trajectories():
    front_lines = ax2.plot(task.env.get_xfhist(), task.env.get_yfhist(), 'r')
    back_lines = ax2.plot(task.env.get_xbhist(), task.env.get_ybhist(), 'b')
    plt.axis('equal')
perform_cumrewards = []
for iteration in range(100000):
    #print("ITERATION :  " , iteration)
    r = exp.doEpisodes(1)
    cumreward = exp.task.getTotalReward()
    #print 'cumreward: %.4f; nsteps: %i; learningRate: %.4f' % (cumreward, len(r[0]), exp.agent.learner.learningRate)
    if iteration % 15 == 0:
        exp.agent = max_agent
        r = exp.doEpisodes(1)
        perform_cumreward = task.getTotalReward()
        perform_cumrewards.append(perform_cumreward)
        print('PERFORMANCE: cumreward:', perform_cumreward, 'nsteps:', len(r[0]))
        stats = (task.env.get_yfhist())
        new_stats = [np.max(stats),np.mean(stats),np.median(stats),stats[-1], perform_cumreward , iteration, exp.agent.learner.learningRate]
        new_stats = [str(s) for s in new_stats]
        with open("res/lspi_30-50.txt", "a") as myfile:
            myfile.write(" ".join(new_stats)+"\n")

        max_agent.reset()
        exp.agent = agent
        ax1.cla()
        ax1.plot(perform_cumrewards, '.--')
        # Wheel trajectories.
        update_wheel_trajectories()
        plt.pause(0.001)
