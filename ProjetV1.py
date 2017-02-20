# coding: utf-8

# In[1]:

import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from matplotlib import pyplot as plt
from copy import deepcopy
#get_ipython().magic(u'matplotlib qt')


# In[ ]:

#import the environnement
from bicycle_env import BicycleEnvironment
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPRegressor


# In[ ]:

env = BicycleEnvironment()


# In[ ]:

#basic constandts :
model_name = "sgd"
nb_actions = 9
nb_states = 10 #for the begguging

scaler = sklearn.preprocessing.StandardScaler()
env.reset()
scaler.partial_fit([env.sensors[:5]])
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform([env.sensors[:5]]))
#scaling ?
scaling = False
new_features = True


# In[ ]:

# basic function to approximate the state value :
#one model for each action
class Approximator():
    def __init__(self,function_type):
        self.models = []
        self.alls  = [[0,0,0,0,0]]
        for i in range(nb_actions):
            model = function_type()
            env.reset()
            #only takes the angles
            X_train  = env.sensors[:5]
            model.partial_fit([self.space_discrete(X_train)],[[0]])
            self.models.append(model)
        self.alls = []
    def space_discrete(self,X):
        if scaling:
            '''scaler.partial_fit([X])
            scaled = scaler.transform([X])
            self.alls.append(X)
            featurizer.fit(scaler.transform(self.alls))'''
            self.alls.append(X)
            scaled = scaler.transform([X])
            featurized = featurizer.transform(scaled)
            return featurized[0]
        else:
            if new_features:
                x_2 = np.power(X[2],2)
                x_0 = np.power(X[0],2)
                res = [1,X[2],X[3],x_2,np.power(X[3],2),X[2]*X[3],X[0],X[1],x_0,np.power(X[1],2),
                      X[0]*X[1],X[2]*X[0],X[2]*x_0,x_2*X[0]]
                return res
            return X
    def predict_values(self, s, a=None):
        if a != None:
            return self.models[a].predict(self.space_discrete(s))
        return [self.models[a].predict([self.space_discrete(s)]) for a in range(len(self.models))]
    def update(self, s, a, y):
        a_int = act_to_int(a)
        self.models[a_int].partial_fit([self.space_discrete(s)],[a_int])
    def retrain_scaler(self):
        scaler.fit(self.alls)
        featurizer.fit(scaler.transform(self.alls))

f = ()
if model_name == "svm":
    f = type(SVR())
elif model_name== "sgd":
    f = type(SGDRegressor())
a = Approximator(f)


# In[ ]:

#metric choice :
metric_id = 1
max_tilt = np.pi / 15.
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    tmp =  np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return np.deg2rad((tmp + 180 + 360) % 360 - 180)
def get_reward(state):
    if metric_id ==0:
        if np.abs(env.getTilt()) > max_tilt:
            return -1.0
        return 0.0
    elif metric_id ==1:
        #chute du velo
        if np.abs(env.getTilt()) > max_tilt:
            return -1.0
        #cible atteinte
        elif state[6]>100:
            return 0.001
        else:
            #rien de special
            tmp = angle_between((state[5],state[6]),(0,100))
            return (4-np.power(tmp,2))*0.00004

def int_to_act(number_action):
    a1 = (number_action/3)
    a2 = (number_action%3)
    return (a1,a2)
def act_to_int(act):
    return act[0] *3 +act[1]
def epsilon_greedy_policy(estimated_reward,epsilon):
    if  np.random.random() > epsilon:
        #exploitation
        best = np.argmax(estimated_reward)
        return int_to_act(best)
    else:
        #exploration :
        return np.random.randint(3,size=2)
def update_wheel_trajectories(ax2):
    #code from the pybrain example
    front_lines = ax2.plot(env.get_xfhist(), env.get_yfhist(), 'r')
    back_lines = ax2.plot(env.get_xbhist(), env.get_ybhist(), 'b')
    plt.axis('equal')
def to_action(action):
    bar_t = [-2.0,0,2.0]#[-1.0,0,1.0]
    speed = [-0.02,0,0.02]
    return [bar_t[action[0]],bar_t[action[1]]]
def sarsa_lambda(env,approx,graphics = False,nb_episodes=10):
    discount_factor = 1.0
    rewards_cumules = []
    if graphics:
        env.saveWheelContactTrajectories(True)
        plt.ion()
        fif = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    should_show = False
    K = 2000
    for ep in range(nb_episodes):
        #reset the environnment
        env.reset()
        #initialize the traces:
        traces  = np.zeros((nb_states,nb_actions))
        #next action
        a_next = np.random.randint(3,size=2)
        #time to learning
        t = 0
        reward_cuml = 0
        epsilon = 1.0
        #current state
        current_state = env.sensors[:5]
        if (ep+1) % 50 == 0:
            should_show = True
            epsilon = 0.0
        else:
            should_show = False
            epsilon = 1.0
        while True:
            if env.time_step>0:
                if a_next != None:
                    action = a_next
                #make the action
                env.performAction(to_action(action))
                reward = get_reward(env.sensors)
                #estimation
                next_values = approx.predict_values(env.sensors[:5])
                #next action
                a_next = epsilon_greedy_policy(next_values,epsilon)
                estimated_value =  reward + discount_factor * next_values[act_to_int(a_next)]
                approx.update(current_state, action, estimated_value)
                reward_cuml += reward * np.power(0.8,t)
                epsilon *= 0.985
                current_state = env.sensors[:5]
            if reward == -1.0 or reward==0.001:
                #discount_factor = (K)/float((K+ep))
                #print("Distance parcourue jusqu'a chute  : " , env.sensors[6] , t)
                if should_show:
                    rewards_cumules.append(reward_cuml)
                if False:
                    approx.retrain_scaler()
                break
            t += 1
        if graphics and should_show:
            print(env.sensors[6])
            ax1.cla()
            ax1.plot(rewards_cumules, '.--')
            update_wheel_trajectories(ax2)
            plt.pause(0.001)
#sarsa_lambda(env,Approximator(f),nb_episodes = 5100, graphics=True)


def discrtisation_etats(etats):
    angle_bar = [-np.pi/2.0,-1.0,-0.2,0,0.2,1.0,np.pi/2.0]
    angular_vecolity = [-np.inf,-2.0,0,2.0,np.inf]
    angle_vertical = [-1/15.0*np.pi,-0.15,-0.06,0,0.06,0.15,1/15.0*np.pi]
    angular_vecolity_d = [-np.inf,-0.5,-0.25,0,0.25,0.5,np.inf]
    angular_accel = [-np.inf,-2.0,0,2.0,np.inf]
    entier = [angle_bar,angular_vecolity,angle_vertical,angular_vecolity_d,angular_accel]
    arr = []
    for i in range(len(etats)):
        arr.append(np.digitize([etats[i]], entier[i]))
    arr = np.array(arr)
    arr -= 1
    return arr.reshape(1,-1)[0]

'''    print("Nombre d'états : ",(len(angle_bar)-1)*(len(angular_vecolity)-1)*(len(angle_vertical)-1)* (len(angular_vecolity_d)-1)*(len(angular_accel)-1))
    indices_discretsation = []
print(discrtisation_etats([-1.1,0,0,0,0]))


test = np.cumprod([1] + [len([1,2,3,4,6,7])-1,len([5,6,3,2])-1, len([1,2,3])-1,len([4,5,6,4])-1])[:-1]
print([1]  + [len([1,2,3,4,6,7])-1,len([5,6,3,2])-1, len([1,2,3])-1,len([4,5,6,4])-1])
print("Test : ",test,len([1,2,3,4,6,7]))
'''
