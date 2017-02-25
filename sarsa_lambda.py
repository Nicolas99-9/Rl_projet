import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from matplotlib import pyplot as plt
from copy import deepcopy
import scipy

from bicycle_env import BicycleEnvironment
import seaborn as sn
from reward import getReward_got_to,getReward_got_to_full,getReward_complex,getReward_direction

#creation de l'environnement
env = BicycleEnvironment()
env.reset()


#parametres de base
nb_actions = 9
nb_states = 3456 #for the begguging
#critere d'arret
max_tilt = np.pi / 6.



#discretisation
#------------------------------------------------
angle_bar =[-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi]
angular_vecolity = [-np.inf,-2.0,0,2.0,np.inf]
angle_vertical = [-np.pi / 6., -0.15, -0.06, 0, 0.06, 0.15,np.pi / 6.]
angular_vecolity_d = [-np.inf,-0.5,-0.25,0,0.25,0.5,np.inf]
angular_accel = [-np.inf,-2.0,0,2.0,np.inf]
entier = [angle_bar,angular_vecolity,angle_vertical,angular_vecolity_d,angular_accel]
state_conversion = []
to_int = [ 1 ,6,24,144,864]


#discretise un etat
def discrtisation_etats(etats):
    arr = []
    for i in range(len(etats)):
        arr.append(np.digitize([etats[i]], entier[i]))
    arr = np.array(arr)
    arr = arr-  1
    val =  np.array(arr.reshape(1,-1)[0]).astype(int)
    pred = np.sum([val[i] * to_int[i] for i in range(len(val))])
    tab =  np.zeros(3456)
    tab[pred]=1
    return tab


#angle entre deux points
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    tmp =  np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return np.deg2rad((tmp + 180 + 360) % 360 - 180)

#renvoie la reward en fonction de l'etat actuel
def get_reward(state,metric_id):
    if metric_id ==0:
        if np.abs(env.getTilt()) > max_tilt:
            return -1.0
        return 0.0
    elif metric_id ==1:
        #chute du velo
        if np.abs(env.getTilt()) > max_tilt:
            return -1.0
        else:
            '''#rien de special
            tmp = angle_between((state[5],state[6]),(0,100))
            return (4-np.power(tmp,2))*0.00004'''
            return 0.0
    elif metric_id==2:
        return getReward_complex(env,state)
    elif metric_id==3:
        return getReward_got_to_full(env,state)
    elif metric_id==4:
        return getReward_got_to(env,state)
    elif metric_id==5:
        return getReward_direction(env,state)

#transform la valeur d'une action en un tuple correspondant au deux valeurs d'action
def int_to_act(number_action):
    a1 = (number_action/3)
    a2 = (number_action%3)
    return (a1,a2)

#convrtti une actio en un entier
def act_to_int(act):
    return act[0] *3 +act[1]

#epsilon greedy policy
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

#converti une action en une valeur
def to_action(action,rewardType):
    if rewardType==3:
        speed = [-2.0,0,2.0]#[-1.0,0,1.0]
        bar_t = [-0.02,0,0.02]
        p = 2.0 * np.random.rand() - 1.0
        return [speed[action[0]],speed[action[1]]+ 0.02 * p]
    else:
        speed = [-2.0,0,2.0]#[-1.0,0,1.0]
        bar_t = [-0.02,0,0.02]
        p = 2.0 * np.random.rand() - 1.0
        return [speed[action[0]],bar_t[action[1]]+ 0.02 * p]

expl_proportion = 0.5
#renvoie une action en utilsiant la boltzman policy
def get_action(state,expl_proportion,greedy):
    if not greedy:
        max_val = np.max(state)
        proba = state - max_val
        idx = proba > 0
        proba[idx] = 0
        idx = proba < -15
        proba[idx] = -15
        proba =np.exp(proba)
        #normalization
        proba = proba/float(np.sum(proba))
        return np.random.choice(9,p=proba)
    else:
        arr = np.zeros(nb_actions)
        arr[state==np.max(state)] = 1.0
        pb =  arr* (1 - expl_proportion)+ (expl_proportion / 9.0)
        return np.random.choice(9,p=pb)

def get_action_greedy(state):
    arr = np.zeros(nb_actions)
    arr[state==np.max(state)] = 1.0
    arr = arr/float(np.sum(arr))
    return np.random.choice(9,p=arr)

#code provenant de balanced task
def should_stoe(env,t):
    if np.abs(env.getTilt()) > max_tilt:
        return True
    if env.time_step * t > 1000.0:
        return True
    return False

def get_arr(nb,size):
    tmp = np.zeros(size)
    tmp[nb] = 1.0
    return tmp

def sarsa_lambda(env,graphics = False,nb_episodes=10,metric_id=1,greedy=False,sarsa_normal=False, i = 1):
    file_name = "res/"+str(metric_id) +"_"+str(greedy)+"_new.txt"
    image_name = "res/"+str(metric_id) +"_"+str(greedy)+"_" + str(i)+".png"

    rewards_cumules = []
    global_stats = []
    env.saveWheelContactTrajectories(True)
    if graphics:
        plt.ion()
        fif = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    should_show = False
    lambda_ = 0.95
    discount = 0.99
    weights =  np.random.randn(nb_actions, nb_states) / 10.
    K = 3000
    target_reacehed = False
    #sauvegare des etats actions du passe
    learning_rate  = 0.5
    last_obs = None
    last_action = None
    last_reward = 0.0
    tab = None
    expl_proportion = 0.5
    print(" NEW TEST , metric_id" , metric_id , " greedy " , greedy )
    for ep in range(nb_episodes):
        #reset the environnment
        env.reset()
        traces  = np.zeros((nb_actions,nb_states))
        t = 0
        reward_cuml = 0
        #current state
        if (ep) % 50 == 0:
            should_show = True
        else:
            should_show = False
        learning_rate = learning_rate * (K + ep)/(K + (ep + 1.0))
        expl_proportion *= 0.99
        reward_cuml = 0.0
        while not should_stoe(env,t):
            #get the obersavation
            if should_show:
                #action sans appretissage, uniquement du greedy
                tmp_k = discrtisation_etats(env.getSensors()[:5])
                env.performAction(to_action(int_to_act(get_action_greedy(np.dot(weights,tmp_k))),metric_id))
                r = get_reward(env.getSensors(),metric_id)
                reward_cuml += r * np.power(0.99,t)
            else:
                #apprentissage
                copy_last_action = None
                copy_last_obs = None
                obs = discrtisation_etats(env.getSensors()[:5])
                if last_obs != None:
                    copy_last_obs = deepcopy(last_obs)
                    copy_last_action = deepcopy(last_action)
                last_obs = obs
                last_action = get_action(np.dot(weights,obs),expl_proportion,greedy)
                if copy_last_action != None:
                    future = np.dot(weights[last_action],obs)
                    paste = np.dot(weights[copy_last_action],copy_last_obs)
                    target = last_reward + discount * future - paste
                    if sarsa_normal:
                        traces *= discount * lambda_
                        traces[copy_last_action] = traces[copy_last_action]  + copy_last_obs
                    else:
                        traces *= discount * lambda_
                        state_t1 = copy_last_action
                        state_t0 = copy_last_obs
                        actions_discreste = get_arr(state_t1,9)
                        new_trace = traces[state_t1] + (state_t0)
                        idx = new_trace < -np.inf
                        new_trace[idx] = -np.inf
                        idx = new_trace > 1.0
                        new_trace[idx] = 1.0
                        traces[state_t1] = new_trace
                        position = np.where(actions_discreste!=1)[0].reshape(-1,1)
                        traces[position,np.where((state_t0) == 1)] = 0.0
                    weights +=  learning_rate * target * traces
                #execute l'action
                env.performAction(to_action(int_to_act(last_action),metric_id))
                #recupere la reward
                reward = get_reward(env.getSensors(),metric_id)
                last_reward = reward
            t+=1
        if should_show:
            stats = (env.get_yfhist())
            print 'iteration : %i; cumreward: %.4f; nsteps: %i; learningRate: %.4f; max distance : %.4f' % (ep/50,reward_cuml, t, learning_rate, np.max(stats))
            new_stats = [np.max(stats),np.mean(stats),np.median(stats),stats[-1], reward_cuml , t, learning_rate]
            new_stats = [str(s) for s in new_stats]
            with open(file_name, "a") as myfile:
                myfile.write(" ".join(new_stats)+"\n")
            global_stats.append(new_stats)
            print(env.sensors[6])
            if graphics:
                ax1.cla()
                rewards_cumules.append(reward_cuml)
                ax1.plot(rewards_cumules, '.--')
                update_wheel_trajectories(ax2)
                plt.pause(0.001)
                if len(rewards_cumules)%50 ==0:
                    plt.savefig(image_name +".png")
    plt.savefig(image_name+".png")
    return global_stats
total_stats = {}
#sarsa_lambda(env,nb_episodes = 100, graphics=True,metric_id=3,greedy =False)
nb_test = 2
import pickle
for g in [False]:
    for nb in [4,5]:# [1,2,3,4,5]
        cle = (g,nb)
        tmp_stats = []
        print("CLE : ", cle)
        for i in range(nb_test):
            print("Iteration !: ", i)
            file_name = "res/"+str(nb) +"_"+str(g)+"_new.txt"
            with open(file_name, "a") as myfile:
                myfile.write("DEBUT ITERATION : " + str(i) + "\n")
            tmp_stats.append(sarsa_lambda(env,nb_episodes = 20000, graphics=True,metric_id=nb,greedy =g, i = i))

'''print("Nombre d'etats : ",(len(angle_bar)-1)*(len(angular_vecolity)-1)*(len(angle_vertical)-1)* (len(angular_vecolity_d)-1)*(len(angular_accel)-1))
indices_discretsation = []
print(discrtisation_etats([-1.1,0,0,0,0]))


test = np.cumprod([1] + [len([1,2,3,4,6,7])-1,len([5,6,3,2])-1, len([1,2,3])-1,len([4,5,6,4])-1])[:-1]
print([1]  + [len([1,2,3,4,6,7])-1,len([5,6,3,2])-1, len([1,2,3])-1,len([4,5,6,4])-1])
print("Test : ",test,len([1,2,3,4,6,7]))
'''
