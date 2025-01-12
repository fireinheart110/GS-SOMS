#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from time import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Sobol import i4_sobol_generate
import GPy
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from pyswarm import pso
# In[2]:


def STOCHOS(norm_power,
            norm_powerL,
            el_price,
            el_priceL,
            op_time,
            op_timeL,
            rle,
            access_LTH,
            theta=None,
            rho=None,
            curtailment=0,
            scalars=None,
            STH_sched=None,
            return_all_vars=False,
            rel_gap = 0.01,
            CMS = False,
            TBS = False
           ):
    '''
    STOCHOS -> HOST 的随机实现
    
    Inputs:
    - norm_power: np.array of shape (STH_dim,N_t,N_S) -> 一个三维数组，STH中风力涡轮机的每小时归一化电力情景。
    - norm_powerL: np.array of shape (LTH_dim,N_t,S) -> 一个三维数组，LTH中风力涡轮机的每日归一化功率情景。
    - el_price: np.array of shape (STH_dim,N_S) -> 一个二维数组，STH地区每小时电价情景[美元/兆瓦时]
    - el_priceL: np.array of shape (LTH_dim,N_S) -> 一个二维数组，LTH的每日电力价格情景[美元/兆瓦时]
    - op_time: np.array of shape (STH_dim,N_t,N_S) -> 一个三维数组，适用于涡轮机维护的小时运行时间方案[小时]
    - op_timeL: np.array of shape (LTH_dim,N_t,N_S) -> 一个三维数组，涡轮机维护的每日操作时间方案[小时]
    - rle: np.array of shape (N_t,N_S) -> 一个二维数组，每个风力涡轮的剩余寿命估计场景[天]
    - access_LTH: np.array of shape (LTH_dim,N_t,N_S) -> 一个三维数组，每天涡轮机维护的运行时间场景[小时]
    - theta: (optional) np.array of shape (N_t,) -> 一个一维数组，二进制参数，表示风力涡轮机需要维护。
        默认情况下，所有涡轮机都需要维护。. theta = np.ones(N_t)
    - rho: (optional) np.array of shape (N_t,) ->  一个一维数组，二进制参数，表示过去是否进行过维护。。
        默认情况下，过去没有进行维护。. rho = np.zeros(N_t)
    - curtailment: (optional) np.array of shape (STH_dim,N_S) -> 一个标量，每小时功率削减（归一化）。默认值为零。
    - scalars:  一个元组，(optional) tuple or list containing scalar parameter values with the following order:
        Kappa (default: 4000) 每个PM任务的成本 [$]
        Fi (default: 10000) 每个CM任务的成本 [$]
        Psi (default: 250) 维护人员的每小时成本[美元/小时]
        Omega (default: 2500) 船只每日租金[美元/天]
        Q (default: 125) 维护人员加班费用[美元/小时]
        R (default: 12) 风力涡轮机额定功率输出[兆瓦]
        W (default: 8) 标准支付工作小时数 [小时]
        B (default: 2) 维护人员数量
        H (default: 8) 总加班小时数 [小时] 的最大值
        tR (default: 5) Time of first sunlight
        tD (default: 21) Time of last sunlight
    - STH_sched: (optional) -> 一个字典，包含固定的日前PM计划的pd.DataFrame列表，如果用户想要模拟日前。在这种情况下，应使用真实参数值作为确定性数据点（即N_S=1）。
        Default value is None. 
    - return_all_vars: (optional) bool -> 一个布尔值，返回整个优化模型，而不是选择的处理后的输出。
        默认值为False以最小化内存使用。
    - rel_gap: (optional) float -> 一个标量，设置优化器的相对最优间隙。
        
    Returns:
    - output: 一个字典（仅当return_all_vars设置为False时），具有以下键： 
        'STH_PM_sched': pd.DataFrame of shape (STH_dim,N_t) with the PM schedule of the STH.
        'STH_CM_sched': pd.DataFrame of shape (STH_dim,N_t) with the CM schedule of the STH. 
        'LTH_PM_sched': np.array of shape (STH_dim,N_t,N_S) with the stochastic PM schedule of the LTH. 
        'LTH_CM_sched': np.array of shape (STH_dim,N_t,N_S) with the stochastic CM schedule of the LTH.
        'expected_STH_cost': STH的预期成本。
        'expected_LTH_cost': LTH的预期成本。
        'total_expected_cost': 总预期成本，即STH和LTH成本的总和，加上STH中未完成维护任务产生的预期成本。
        'remaining_maint_hrs': np.array of shape (N_t, N_S) 对于在STH中启动但在同一天未完成的任务，剩余的维护小时数。
        'model': 无，或包含优化模型所有变量值的字典（仅在return_all_vars设置为True时）。
    '''
    
    ## 构造集合 ##
    tF, N_t, N_S = norm_power.shape
    dJ = norm_powerL.shape[0]
    
    I = ['wt'+str(i) for i in range(1,N_t+1)]
    T = ['t'+str(t) for t in range(tF)]
    D = ['d'+str(d) for d in range(1,dJ+1)]
    S = ['s'+str(s) for s in range(1,1+N_S)]
    
    
    ## 指定默认参数 ##
    rho = pd.Series(np.zeros(N_t), index = I) if rho is None else pd.Series(rho, index=I) #全零向量
    theta = pd.Series(np.ones(N_t), index = I) if theta is None else pd.Series(theta, index=I) #全一向量
    
    C = pd.DataFrame(np.ones((tF,N_S))-curtailment, columns=S, index=T).T #一个二维数组，表示功率输出的限制系数
    
    # 标量参数和成本
    if scalars is None:
        Kappa = 4000      #($/PM task) 每个PM任务的成本
        Fi = 10000        #($/CM task) 每个CM任务的成本
        Psi = 250         #($/h) 维修人员小时费用
        Omega = 2500      #($/d) 船舶日租金
        Q = 125           #($/h) 维修人员加班费
        R = 12            #(MW) 风力发电机的额定输出功率
        W = 8             #(ts) 标准付款的工程数量
        B = 2             #(-) 维修人员数量
        H = 8
        tR = 5   
        tD = 21
    else:
        Kappa,Fi,Psi,Omega,Q,R,W,B,H,tR,tD = scalars
    
    Qa = 5000
    
    # 附加参数
    rle = rle.copy()
    
    zeta = np.ones((N_t))
    zeta[np.all(rle==0, axis=1)] = 0 
    zeta = pd.Series(zeta, index=I) #zeta: 一个一维数组，表示每个风机是否需要维修，如果RLE为零，则为0；否则为1。

    zetaLong = np.ones((dJ,N_t,N_S))
    zetaLong[np.repeat(rle[np.newaxis,:,:],dJ,0)-np.arange(dJ).reshape((-1,1,1)) <= 1] = 0 #zetaLong: 一个三维数组，表示LTH下每个风机每天每个情景是否需要维修，如果RLE小于等于当天的序号，则为0；否则为1。
    #print(rle[0,:])
    #print(zetaLong[:,0,:])
    zetaL = {d: {i: {s: zetaLong[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    
    ## 将随机参数转换为适当的格式（字典） ##
    Pi = {t:{s: el_price[ti,si] for si, s in enumerate(S)} for ti, t in enumerate(T) }
    PiL = {d:{s: el_price[di,si] for si, s in enumerate(S)} for di, d in enumerate(D) }
    norm_power2 = norm_power.copy()
    norm_power2[norm_power2<=1e-4]=1e-4
    f = {t: {i: {s: norm_power2[ti,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for ti, t in enumerate(T)}
    fL = {d: {i: {s: norm_powerL[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    A = {t: {i: {s: op_time[ti,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for ti, t in enumerate(T)}
    AL = {d: {i: {s: op_timeL[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    
    #mission_time = repair_time + {frac of repair time that will be inaccessible}
    #mission_time = repair_time + repair_time*(1-access)
    A_LTH = {d: {i: {s: (1 + (1-access_LTH[di,ii,si])) for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}

    
    
    ## 初始化模型实例并声明变量 ##
    owf = gp.Model('STOCHOS')
    owf.update()

    # 连续变量
    p = owf.addVars(T, I, S, lb=0, name="p") # 在STH中的小时功率输出
    pL = owf.addVars(D, I, S, lb=0, name="pL") # 在LTH中的每天功率输出
    l_STH = owf.addVar(lb=-np.inf, name = "l_STH") #在STH中获得的利润
    l_LTH = owf.addVars(D, lb=-np.inf, name = "l_LTH") #在LTH的第d天获得的利润

    # 整数变量
    q = owf.addVars(S,lb = 0, vtype = gp.GRB.INTEGER, name = "q") # 加班时长
    qL = owf.addVars(D, S, lb = 0, vtype = gp.GRB.INTEGER, name = "qL") # 加班时长
    qa = owf.addVars(S,lb = 0, vtype = gp.GRB.INTEGER, name = "qa") # 加班时长
    xa = owf.addVars(S,lb = 0, vtype = gp.GRB.INTEGER, name = "xa") # 加班时长
    b = owf.addVars(I, S, lb = 0, vtype = gp.GRB.INTEGER, name = "b") # 加班时长
    
    

    # 二元变量
    m = owf.addVars(T, I, vtype = gp.GRB.BINARY, name = "m") 
    mL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "mL") 
    y = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "y")
    yL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "yL")
    v = owf.addVar(vtype = gp.GRB.BINARY, name = "v")
    vL = owf.addVars(D, S, vtype = gp.GRB.BINARY, name = "vL")
    x = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "x")
    w = owf.addVars(I, S, vtype = gp.GRB.BINARY, name = "w")
    z = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "z")  #u
    
    ## Simulation mode ##
    if STH_sched is not(None):
        owf.reset(0)
        print("*** Initiating day-ahead simulation from input schedule ***")
        for i in I:
            for t in T:

                if STH_sched[i][t]==1:
                    m[t,i].lb=1
                else:
                    m[t,i].ub=0
                    
    else:
        print("#########################################################")
        print("# STOCHOS -> Stochastic Holistic Opportunistic Strategy #")
        print("#########################################################")


    #owf.update()



    U = {i: {s: Omega+R*PiL[D[0]][s]*fL[D[0]][i][s]*tR for s in S}
         for i in I} #未完成已启动任务的前期成本
    Y = {i: {s: R*PiL[D[0]][s]*fL[D[0]][i][s] for s in S} 
         for i in I} #未完成已启动任务的每小时成本

    
    # 约束条件
    obj_fun = l_STH + gp.quicksum(l_LTH[d] for d in D) - 1/N_S*(gp.quicksum(
        U[i][s]*w[i,s]+Y[i][s]*b[i,s]*A_LTH['d1'][i][s] for i in I for s in S) + gp.quicksum(Qa*(xa[s]+qa[s]) for s in S))  #目标函数

    con2 = owf.addConstr((l_STH == -gp.quicksum((1-rho[i])*(Kappa+(1-zeta[i])*(Fi-Kappa))*m[t,i]
                                               for t in T for i in I)-Omega*v+
                          1/N_S*gp.quicksum( gp.quicksum(Pi[t][s]*p[t,i,s]-Psi*x[t,i,s] 
                                                        for t in T for i in I)
                                            -Q*q[s] for s in S)), name = "STH profit")

    con3 = owf.addConstrs((l_LTH[d] == 1/N_S*
                           gp.quicksum(gp.quicksum(PiL[d][s]*pL[d,i,s]-(1.-rho[i])*(
                               Kappa+(1-zetaL[d][i][s])*(Fi-Kappa))*mL[d,i,s]-Psi*AL[d][i][s]*(
                               mL[d,i,s]) for i in I)-Omega*vL[d,s]-Q*qL[d,s] for s in S) for d in D), 
                          name = "LTH profit")

    con4 = owf.addConstrs((gp.quicksum(m[t,i] for t in T)+
                          gp.quicksum(mL[d,i,s] for d in D) == 
                          theta[i] for i in I for s in S), name = "Force maintenance")

    con5 = owf.addConstrs((m[t,i] <= (T.index(t))/tR for t in T for i in I), 
                          name = "Maintenance after sunrise")

    con6 = owf.addConstrs((m[t,i] <= tD/(1.01+T.index(t)) for t in T for i in I), 
                          name = "Maintenance before sunset")

    con7 = owf.addConstrs((gp.quicksum(z[T[T.index(t)+t_hat],i,s] 
                                       for t_hat in range(min([len(T[T.index(t):]),A[t][i][s]])))
                          >= min([len(T[T.index(t):]),A[t][i][s]])*m[t,i] 
                           for t in T for i in I for s in S
                          if T.index(t)+1.01 <= tD), name = "Downtime from ongoing maintenance")

    con8 = owf.addConstrs((b[i,s] >= gp.quicksum(m[t,i]*max([0,A[t][i][s]-len(T[T.index(t):])]) 
                                                    for t in T) for i in I for s in S), 
                          name = "Remaining hours of unfinished maintenance")

    con9 = owf.addConstrs((w[i,s] >= b[i,s]/100 for i in I for s in S), name = "Unfinished maintenance")

    con10 = owf.addConstrs((x[t,i,s] >= z[t,i,s]-(T.index(t))/tD for t in T for i in I for s in S), 
                           name = "Crew occupacy")

    con11 = owf.addConstrs((gp.quicksum(x[t,i,s] for i in I) <= B + xa[s]  for t in T for s in S), 
                           name = 'Max tasks per t')    #+ ba[s]<------------------------------

    con16 = owf.addConstrs((y[t,i,s] <= zeta[i]*(1-rho[i])
                           + gp.quicksum((len(T[T.index(tt):]))*m[tt,i] for tt in T)/
                             (len(T[T.index(t):])+0.1) + 1.0-theta[i]
                             for t in T for i in I for s in S), name = 'Availability STH')

    con17 = owf.addConstrs((yL[d,i,s] <= zetaL[d][i][s]*(1-rho[i])
                            +(dJ-gp.quicksum((D.index(dd)+1)*mL[dd,i,s] for dd in D))/
                            (dJ-D.index(d)-0.9) +(1.0-theta[i]) for d in D for i in I for s in S),
                            name = "Availability LTH")

    con18 = owf.addConstrs((y[t,i,s] <= 1 - z[t,i,s] for t in T for i in I for s in S),
                           name = "Unavailability from maintenance")

    con19 = owf.addConstr((v>=1/N_t*gp.quicksum(m[t,i] for t in T for i in I)), 
                           name = "STH vessel rental")

    con20 = owf.addConstrs((vL[d,s] >= 1/N_t*gp.quicksum(mL[d,i,s] for i in I) 
                            for d in D for s in S), name = "LTH vessel rental")

    con21 = owf.addConstrs((gp.quicksum(x[t,i,s] for t in T for i in I)<=B*W+q[s]+qa[s] for s in S), 
                          name = "Overtime")

    con22 = owf.addConstrs((gp.quicksum(mL[d,i,s]*AL[d][i][s]+b[i,s] for i in I) 
                            <= B*W+qL[d,s] for d in D for s in S if D.index(d)==0), 
                           name = 'Overtime 1st day of LTH')

    con23 = owf.addConstrs((gp.quicksum(mL[d,i,s]*AL[d][i][s] for i in I) 
                            <= B*W+qL[d,s] for d in D for s in S if D.index(d)>0), 
                           name = 'Overtime other days of LTH')

    con24 = owf.addConstrs((q[s]<=H for s in S), name = "STH max overtime")

    con25 = owf.addConstrs((qL[d,s]<=H for d in D for s in S), name = "LTH max overtime")

    con26 = owf.addConstrs((p[t,i,s]<=R*(f[t][i][s])*y[t,i,s] 
                            for t in T for i in I for s in S),
                           name = "STH power")

    con27 = owf.addConstrs((pL[d,i,s]<=24*R*(fL[d][i][s])*(yL[d,i,s]-mL[d,i,s]*zetaL[d][i][s]*AL[d][i][s]/24) 
                            for d in D for i in I for s in S),name = "LTH power")

    con28 = owf.addConstrs((gp.quicksum(p[t,i,s] for i in I) <= 
                           gp.quicksum(f[t][i][s] for i in I)*R*C[t][s] for t in T for s in S), 
                           name = 'Power curtailment')
    
    if CMS:
        cms_con = owf.addConstrs((gp.quicksum(m[t,i] for t in T) <= 1-zeta[i] for i in I), 
                           name = 'CMS benchmark constraint')
    
    if TBS:
        assert len(S)==1, 'TBS is a deterministic benchmark.'
        tbs_con = owf.addConstrs((gp.quicksum(m[t,i] for t in T) == theta[i]-zetaL['d1'][i]['s1'] for i in I), 
                           name = 'TBS benchmark constraint')
        

    #########################################################################################################


    # 设定目标
    owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

    owf.setParam("MIPGap", rel_gap)
    owf.setParam("TimeLimit", 3600)

    owf.update()
    

    # 求解模型
    owf.optimize()
    
    
    if owf.solCount == 0:
        return None
    else:

        STH_sched = pd.DataFrame(np.round(np.array(list((owf.getAttr('X',m).values()))).reshape(-1,len(I))), 
                                 columns = I, index = T)

        LTH_sched = np.array(list(owf.getAttr('X',mL).values())).reshape(dJ,N_t,N_S)

        expected_STH_cost = (1/N_S*np.sum([f[t][i][s]*R*Pi[t][s]*C[t][s] for t in T for i in I for s in S])-l_STH.X)
        expected_LTH_cost = (1/N_S*np.sum([24*fL[d][i][s]*R*PiL[d][s] for d in D for i in I for s in S])-
                             np.sum(list(owf.getAttr('X',l_LTH).values())))
        total_expected_cost = expected_STH_cost + expected_LTH_cost + 1/N_S*np.sum([U[i][s]*owf.getAttr('X',w)[i,s]+
                                                                                  Y[i][s]*owf.getAttr('X',b)[i,s] 
                                                                                  for i in I for s in S])

        output = {'STH_sched': STH_sched,   
                  'LTH_sched': LTH_sched, 
                  'expected_STH_cost': expected_STH_cost,
                  'expected_LTH_cost': expected_LTH_cost,
                  'total_expected_cost': total_expected_cost,
                  'remaining_maint_hrs': np.array(list(owf.getAttr('X',b).values())).reshape(N_t,N_S),
                  'model': {var.varName: owf.getVarByName(var.varName).X for var in owf.getVars()} if return_all_vars else None
                 }
        return output


# In[3]:


def hourly2daily_aggregation(data):
    """
    将小时数据转换为日数据，每天的数据取平均值
    
    Input:
    - data: np.array of shape (hours, N_t, N_S)
    
    Returns:
    - output: np.array of shape (hours//24, N_t, N_S) with 输入数组的每日平均值
    """
    total_days = data.shape[0]//24
    output = np.array([np.mean(data[24*x:24*(x+1)],0) for x in range(total_days)])
    return output


# In[4]:

class RidgeRegression():
    
    def __init__(self, reg=0.001):
        """
        一个带有正则化参数由用户定义的岭回归模型（默认值为0.001），这个参数可以控制正则项的强度，越大则惩罚越强，模型越简单；越小则惩罚越弱，模型越复杂。
        """
        self.reg = reg
    
    def fit(self, X, y):
        """
        返回岭回归系数，以最小化提供的数据的平方误差
        
        Inputs:
        -X: np float array of shape (obs_dim, feat_size)
        -Y: np float array of shape (obs_dim, 1)
        
        Returns:
        -self.b: np floar array of shape (feat_size, 1)
        -train_MAE: float -> 训练集的平均绝对误差
        """
        self.b = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.reg*np.eye(X.shape[-1])),np.matmul(X.T,y))
        train_MAE = np.abs(np.matmul(X,self.b)-y).mean()
        return self.b, train_MAE
    
    def predict(self, X):
        out = np.matmul(X,self.b)
        out[out<0]=0
        out[out>1]=1
        return out


#%%

def quadKernelAugmentedSpace(X):
    """
    将形状为（obs_dim，2）的输入空间映射到二次核的形状为（obs_dim，6）的增强空间。
    
    X的第一列是每日平均风速，第二列是每日平均波高。
    """
    out = np.append(np.sqrt(2)*X,X**2,1)
    out = np.append(out, (np.sqrt(2)*X[:,0]*X[:,1]).reshape(-1,1),1)
    out = np.append(out, np.ones((X.shape[0],1)) ,1)
    return out

#%%

def get_mission_time_LTH(wind_speed_forecast_LTH, wave_height_forecast_LTH, tau, wave_lim=1.5, coefs=None):
    """
    返回LTH每天的预期任务时间小时数日平均风速及浪高
    
    Inputs:
    - wind_speed_forecast_LTH: np float array of shape (dJ, N_t, N_S)
    - wave_height_forecast_LTH: np float array of shape (dJ, N_S)
    - tau: np int array of shape (N_t,)
    - wave_lim: 浪高安全阈值(1.5m、1.8m或2m)采用预先计算的回归参数值。
    - coefs: np float arry of shape (1,6) -> 使用二次核空间的回归系数。 
        如果使用默认值None，则将为wave_lim使用三个预计算值中的一个。
    
    Returns:
    - mission_time: np float array of shape (dJ, N_t, N_S) -> LTH任务时间场景
    - access_LTH: np float array of shape (dJ, N_t, N_S) -> 每日可访问性分数场景
    """
    (dJ, N_t, N_S) = wind_speed_forecast_LTH.shape
    #二次核ridgerregression(0.0001)的增广空间权重
    #使用历史NWP数据和真实可访问性数据，计算一天中LTH中涡轮机不可访问的小时数百分比
    #张贴在这里是为了稍微降低复杂性
    if coefs==None:
        if wave_lim==1.5:
            b = np.array(  [[ 0.02008826],
                            [0.76945388],
                            [-0.00409112],
                            [ 0.06934806],
                            [ 0.02252065],
                            [ 1.68621608]])  
        elif wave_lim==1.8:
            b = np.array(  [[ 0.03513277],
                            [-0.58401405],
                            [-0.00536439],
                            [ 0.02710852],
                            [ 0.02153183],
                            [ 1.47014811]])
        elif wave_lim==2:
            b = np.array(  [[ 4.18107897e-02],
                            [-4.62873380e-01],
                            [-6.23088680e-03],
                            [-7.37855258e-04],
                            [ 2.19760977e-02],
                            [ 1.32768890e+00]])
    else: b=coefs
    
    mission_time = np.zeros((dJ, N_t, N_S))
    access_LTH = np.zeros((dJ, N_t, N_S))
    for i in range(N_t):
        for s in range(N_S):
            X_in = np.append(wind_speed_forecast_LTH[:,i,s].reshape(-1,1),
                             wave_height_forecast_LTH[:,s].reshape(-1,1), 1)
            X = quadKernelAugmentedSpace(X_in)
            
            access_LTH[:,i,s] = np.matmul(X.copy(),b).reshape(-1) #(dJ,)
            access_LTH[access_LTH<0] = 0
            access_LTH[access_LTH>1] = 1
            #mission_time = repair_time + {frac of repair time that will be inaccessible}
            #mission_time = repair_time + repair_time*(1-access)
            mission_time[:,i,s] = tau[i]*(1 + (1-access_LTH[:,i,s]))
            
    
    return mission_time, access_LTH
            
#%%
def normal_gen_sobol(size):
    '''
    使用Sobol方法生成均匀样本和极坐标方法作为转换技术，生成从标准正态分布中抽取的随机数序列。
    '''
    U = i4_sobol_generate(2, size)
    U1 = U[:,0]
    U2 = U[:,1]
    D = -2*np.log(U1)
    Theta = 2*np.pi*U2
    return np.sqrt(D)*np.cos(Theta)

def MVN_gen(mean_y, cov_y, N_S):
    '''
    多元正态低差异序列生成器
    '''
    pred_len = mean_y.shape[0]
    # Create L
    
    L = np.linalg.cholesky(cov_y)
    # 用Sobol从标准正态分布中抽取样本X
    X = normal_gen_sobol(size=pred_len*N_S).reshape(-1,N_S)
    # 应用变换
    Y = L.dot(X) + mean_y.reshape(pred_len,1)
    return Y

# def error_scenario_gen(
#         forecast_error_hist,
#         pred_len,
#         N_S,
#         custom_hist_len = 24*10,
#         random_seed=1,
#         LDS = False,
#         DGP=False
#         ):

#     """
#     使用确定性预测误差生成场景 
#     sklearn.gaussian_process.GaussianProcessRegressor fitting and sampling.
    
#     Inputs:
#     - forecast_error_hist: np float array of shape (hist_len,) -> Full error history
#     - pred_len: int -> prediction length
#     - N_S: int -> Number of scenarios to generate
#     - *Optional*
#         - custom_hist_len: int -> Custom history length. If hist_len<custom_hist_len,
#             then hist_len is used (default=24*10)
#         - random_seed=1
        
#     Returns:
#     -forecast_error_scenarios: np float array of shape (pred_len, N_S)
#     """
#     hist_len = forecast_error_hist.shape[0]
    
#     forecast_error_scenarios = np.zeros((pred_len, N_S))
    
#     if custom_hist_len>hist_len: custom_hist_len=hist_len
    
#     kernel = RBF()
#     gp_model = GaussianProcessRegressor(
#             kernel=kernel, 
#             alpha = np.random.uniform(size=(custom_hist_len))**2,
#             random_state=random_seed,  
#             n_restarts_optimizer=10)
    
#     X = np.arange(custom_hist_len).reshape(-1,1)
#     Y = forecast_error_hist[hist_len-custom_hist_len:]
#     gp_model.fit(X, Y)
    
#     X_pred = (X[-1]+np.arange(1,pred_len+1)).reshape(-1,1)
    
#     if np.all(Y==0):
#         forecast_error_scenarios = np.zeros((pred_len, N_S))
#     elif DGP:
#         mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
#         forecast_error_scenarios = mean_y.reshape(-1,1)
#     else:
#         if LDS:
#             mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
#             cov_y = cov_y+np.eye(cov_y.shape[0])*1e-6
#             scenario_set = MVN_gen(mean_y, cov_y, 1000)
#             sorted_std = np.argsort(np.std(scenario_set,0))
#             forecast_error_scenarios = scenario_set[:,sorted_std[:N_S]]
#         else:
#             forecast_error_scenarios = gp_model.sample_y(X_pred,N_S,random_state=random_seed)
        
#     return forecast_error_scenarios

def objective_function(params, input_data, output_data, inputn_test, output_test):
    # 解析参数
    length_scale = params[0]
    noise_level = params[1]
    
    # 创建核函数
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
    
    # 创建GPR模型
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_level**2)
    
    # 训练模型
    X_train = input_data
    y_train = output_data
    
    gpr.fit(X_train, y_train)
    
    # 使用验证集评估模型性能
    X_val = inputn_test
    y_val = output_test
    score = gpr.score(X_val, y_val)
    
    # 返回适应度值（在这里，我们使用模型的平方误差作为适应度值）
    return -score

def error_scenario_gen(
        forecast_error_hist,
        pred_len,
        N_S,
        custom_hist_len = 24*10,
        random_seed=1,
        LDS = False,
        DGP=False,
        GPY=False,
        MAT=False,
        GS=False,
        LSTM=False,
        MLP=False
        ):

    """
    使用确定性预测误差生成场景 
    sklearn.gaussian_process.GaussianProcessRegressor fitting and sampling.
    
    Inputs:
    - forecast_error_hist: np float array of shape (hist_len,) -> Full error history
    - pred_len: int -> prediction length
    - N_S: int -> Number of scenarios to generate
    - *Optional*
        - custom_hist_len: int -> Custom history length. If hist_len<custom_hist_len,
            then hist_len is used (default=24*10)
        - random_seed=1
        
    Returns:
    -forecast_error_scenarios: np float array of shape (pred_len, N_S)
    """
    hist_len = forecast_error_hist.shape[0]
    
    forecast_error_scenarios = np.zeros((pred_len, N_S))

    cut = custom_hist_len
    
    if custom_hist_len>hist_len: custom_hist_len=hist_len

    X = np.arange(custom_hist_len).reshape(-1,1)
    Y = forecast_error_hist[hist_len-custom_hist_len:]
    if LSTM:
        if custom_hist_len<cut:
            kernel = RBF()
            gp_model = GaussianProcessRegressor(
                    kernel=kernel, 
                    alpha = np.random.uniform(size=(custom_hist_len))**2,
                    random_state=random_seed,  
                    n_restarts_optimizer=10)
        else:
            # 定义搜索空间边界
            lb = [1e-40, 1e-6]  # 长尺度参数下限和上限
            ub = [1e20, 1e6]    # 长尺度参数上限和下限

            # 初始化粒子群和速度
            n_particles = 30
            dim = len(lb)
            xopt, fopt = pso(lambda params: objective_function(params, X[:cut-24], Y[:cut-24],X[cut-24:], Y[cut-24:]), lb, ub, swarmsize=n_particles)

            
            # 解析参数
            length_scale = xopt[0]
            noise_level = xopt[1]
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
            
            X_pred = (X[-1]+np.arange(1,pred_len+1)).reshape(-1,1)
            
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise_level**2, random_state=0, n_restarts_optimizer=10)
            gpr.fit(X, Y)
            Y_pred, sigma_pred = gpr.predict(X_pred, return_cov=True)
            forecast_error_scenarios = gpr.sample_y(X_pred,50) #设置要生成的场景数量
    else:
        if GPY:
            # 设置随机种子
            np.random.seed(42)
            # 创建和配置稀疏高斯过程模型
            kernel = GPy.kern.RBF(input_dim=1)
            model = GPy.models.SparseGPRegression(X, Y.reshape(-1, 1), kernel)
            
        elif MAT:
            # 设置随机种子
            np.random.seed(42)
            # 使用Matern核函数
            kernel = Matern(nu=0.5)
            # gp_model = GaussianProcessRegressor(kernel=kernel, alpha=np.random.normal(0.4, 1, (Y.shape[0]))**2,
            #                             random_state=0, n_restarts_optimizer=10)
            gp_model = GaussianProcessRegressor(kernel=kernel, alpha=np.random.uniform(size=(custom_hist_len))**2,
                                        random_state=0, n_restarts_optimizer=10)
        elif GS:
            # 设置随机种子
            np.random.seed(42)
            # 定义参数网格
            param_grid = {
                'kernel': [RBF()],
                'alpha': np.random.uniform(size=(custom_hist_len))**2
            }

            # 使用GridSearchCV进行参数优化，交叉验证优化
            gp_model = GaussianProcessRegressor(random_state=0, n_restarts_optimizer=10)
            grid_search = GridSearchCV(gp_model, param_grid, cv=5)
            grid_search.fit(X, Y)
        
        else:
            kernel = RBF()
            gp_model = GaussianProcessRegressor(
                    kernel=kernel, 
                    alpha = np.random.uniform(size=(custom_hist_len))**2,
                    random_state=random_seed,  
                    n_restarts_optimizer=10)
        
        if GPY:
            # 拟合模型
            model.optimize(messages=True)
        else:
            gp_model.fit(X, Y)
        
        X_pred = (X[-1]+np.arange(1,pred_len+1)).reshape(-1,1)

        if MLP:
            # 节点个数
            # inputnum = 2  # 输入层节点数量
            hiddennum = 5  # 隐含层节点数量
            # outputnum = 1  # 输出层节点数量


            
            net = MLPRegressor(hidden_layer_sizes=(hiddennum,), activation='tanh', solver='lbfgs', max_iter=5000, learning_rate_init=0.01)

            
            net.fit(X, Y)

            
            # inputn_test = scaler_input.transform(input_test.T).T

            
            an = net.predict(X_pred)
        
        if np.all(Y==0):
            forecast_error_scenarios = np.zeros((pred_len, N_S))
        elif DGP:
            mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
            forecast_error_scenarios = mean_y.reshape(-1,1)
        
        else:
            if LDS:
                if MLP:
                    # test_simu = scaler_output.inverse_transform(an.reshape(-1, 1)).flatten()
                    # 蒙特卡洛模拟
                    n_simulations = N_S
                    forecast_error_scenarios = np.zeros((pred_len,n_simulations))

                    for i in range(n_simulations):
                        # 添加噪声到训练数据
                        noise = np.random.normal(0, 0.1, X.shape)
                        net.fit(X + noise, Y)
                        forecast_error_scenarios[:,i] = net.predict(X_val)

                else:
                    if GS:
                        # 使用最佳参数进行预测
                        best_gp = grid_search.best_estimator_
                        mean_y, cov_y = best_gp.predict(X_pred, return_cov=True)
                    
                    elif GPY:
                        # 准备验证数据
                        X_val = (X[-1, 0] + np.arange(pred_len)).reshape(-1, 1)
                        mean_y, cov_y = model.predict_noiseless(X_val, full_cov=True)
                    else:
                        mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
                    cov_y = cov_y+np.eye(cov_y.shape[0])*1e-6
                    scenario_set = MVN_gen(mean_y, cov_y, 1000)
                    sorted_std = np.argsort(np.std(scenario_set,0))
                    forecast_error_scenarios = scenario_set[:,sorted_std[:N_S]]
            else:
                if GS:
                    # 使用最佳参数进行预测
                    best_gp = grid_search.best_estimator_
                    mean_y, cov_y = best_gp.predict(X_pred, return_cov=True)
                    forecast_error_scenarios = best_gp.sample_y(X_pred,N_S,random_state=random_seed)

                elif MLP:
                    # test_simu = scaler_output.inverse_transform(an.reshape(-1, 1)).flatten()
                    forecast_error_scenarios = an.reshape(-1,1)
                elif GPY:
                    # 准备验证数据
                    X_val = (X[-1, 0] + np.arange(pred_len)).reshape(-1, 1)
                    mean_y, cov_y = model.predict_noiseless(X_val, full_cov=True)
                    if cov_y.ndim == 2 and cov_y.shape[0] == cov_y.shape[1]:
                        num_samples = N_S
                        forecast_error_scenarios = np.transpose(np.random.multivariate_normal(mean_y.flatten(), cov_y, num_samples))
                else:
                    forecast_error_scenarios = gp_model.sample_y(X_pred,N_S,random_state=random_seed)
        
    return forecast_error_scenarios


# In[5]:


def el_price_scenario_gen_v1(el_prices_mean, 
                             el_price_std = 4,
                             N_S=100,
                             random_seed=1
                            ):
    
    """
    电力价格正态分布的情景生成。
    
    Inputs:
    - el_prices: np.array of floats of shape (obs_dim,)
    - el_price_std: float or np.array of floats of shape (obs_dim,1)
    """
    np.random.seed(random_seed)
    
    obs_dim = el_prices_mean.shape[0]

    el_prices = np.random.normal(loc = el_prices_mean.reshape(-1,1), scale = el_price_std, 
                                            size = (obs_dim,N_S))
    
    el_prices[el_prices<10] = 10
    
    return el_prices


# In[6]:


def rle_scenario_gen(
        rle_mean, 
        rle_std,
        N_S,
        dist = 'normal',
        random_seed=1
        ):

    """
    使用正态分布生成RLE的场景。
    
    Inputs:
    - rle_mean: np float array of shape (N_t,)
    - rle_std: float or np float array of shape (N_t,)
    -dist: str -> 用于场景生成的分布。有效输入为'正态分布'和'威布尔分布'。
        如果选择了“威布尔分布”，则rle_std用作形状参数，rle_mean用作尺度参数。
    """
    np.random.seed(random_seed)
    N_t = rle_mean.shape[0]
    if not(hasattr(rle_std, '__len__')): rle_std = np.repeat(rle_std, N_t, 0)
    
    if dist.lower() == 'normal':
        rle = np.random.normal(loc = rle_mean.reshape(-1,1), scale = rle_std.reshape(-1,1), 
                                                size = (N_t,N_S))
    elif dist.lower() == 'weibull':
        rle = rle_mean.reshape(-1,1)*np.random.weibull(rle_std.reshape(-1,1), 
                                                size = (N_t,N_S))
        rle[rle_std==0, :] = rle_mean[rle_std==0].reshape(-1,1)
    else: 
        print("Valid RLE pdf's are 'normal' and 'weibull'.")
        
    rle[rle<1] = 1
    
    rle[np.repeat(rle_mean.reshape(-1,1)==0,N_S,-1)] = 0 #如果期望rle为0，那么我们可以知道涡轮机肯定已经失效 
                    
    
    return rle


# In[7]:


def binning_method(wind_speed, bins=None):
    """
    执行分箱方法以计算风速时间序列数据的归一化功率。
    
    Inputs:
    - wind_speed: np.array of floats of shape (24*dJ, N_t, N_S) -> 风速场景
    - bins: (optional) np.array of shape (N_bins, 2) -> 第一列为风速间隔，第二列为相应的归一化功率
        
    Returns:
    - norm_power: np.array of 与包含归一化功率值的wind_speed相同的形状
    """
    if bins is None:
        ws_bin_processed = np.array(pd.read_csv('method_of_bins.csv', usecols = [0,1]))
    else:
        ws_bin_processed = bins
    
    norm_power = np.zeros_like(wind_speed)

    for j in range(ws_bin_processed.shape[0]-1):
        index_true_STH = (wind_speed>=ws_bin_processed[j,0]) & (
            wind_speed<ws_bin_processed[j+1,0])
        norm_power[index_true_STH] = ws_bin_processed[j,1]
    
    return norm_power 


# In[8]:


def get_mission_time_STH(wind_speed,
                        wave_height,
                        vessel_max_wave_height,
                        vessel_max_wind_speed,
                        tau,
                        tR=5,
                        tD=21
                        ):
    
    _,N_t,N_S = wind_speed.shape
    
    # 每小时可访问性
    access = np.zeros_like(wind_speed)
    access[(wind_speed<vessel_max_wind_speed) & (
        np.repeat(wave_height.reshape(-1,1,N_S),N_t,axis=1)<vessel_max_wave_height)] = 1
    access2 = access.reshape(-1,24,N_t,N_S)
    access2[:,:tR,:,:] = 0 #sunrise constraint
    access2[:,tD:,:,:] = 0 #sunset constraint
    #print(np.arange(1,25))
    #print(access2[:,:,2,0])
    
    # 计算操作时间
    op_time = np.zeros((wind_speed.shape), dtype=int)
    op_time2 = op_time.reshape(-1,24,N_t,N_S)
    
    
    for i in range(N_t):
        for t in range(24):
            temp = op_time2[:,t,i,:]
            op_time2[:,t,i,:] = (np.cumsum(access2[:,t:,i,:],1)>=tau[i]).argmax(1)
            temp[temp==0] = (23-t+tau[i]-np.sum(access2[:,t:,i,:],1))[temp==0]
    
    op_time2+=1
    
    return op_time2[0,:,:,:], access2[0,:,:,:]


# In[9]:


def data_loader(ws_STH,  
                ws_LTH,
                wh_STH,
                wh_LTH,
                ep_STH,
                ep_LTH, 
                ws_err_hist,
                wh_err_hist,
                ep_err_hist,
                N_S,
                max_wind, 
                max_wave,
                rle_mean,
                rle_std,
                tau,
                rle_dist='weibull',
                tR=5, 
                tD=21,
                hist_len=5,
                random_seed=1,
                sim_day_ahead=False,
                BESN=False,
                NAIVE=False,
                DGP=False,
                GPY=False,
                MAT=False,
                GS=False,
                LSTM=False,
                MLP=False
               ):
    """
    用于使用预测值准备STOCHOS所需随机输入的函数。在此模型中，我们使用真实的小时数据为STH和LTH生成情景。
    未来的版本将需要使用机器学习方法进行每日预测以生成情景。
    
    Inputs:
    - ws_STH: np.array of shape (24,N_t) -> Hourly wind speed data for each wind turbine in the STH
    - ws_LTH: np.array of shape (dJ,N_t) -> Daily average wind speed data for each wind turbine in the LTH
    - wh_STH: np.array of shape (24,) -> Hourly wave height data for STH
    - wh_LTH: np.array of shape (dJ,) -> Daily average wave height data for LTH
    - ep_STH: np.array of shape (24,) -> Hourly electricity prices data for STH
    - ep_LTH: np.array of shape (dJ,) -> Daily average electricity price data for LTH
    - ws_err_hist: np.array of shape (hist_len,N_t) -> 每台涡轮机历史每小时风速预报误差数据
    - wh_err_hist: np.array of shape (hist_len,) -> 历史每小时浪高预报误差资料
    - ep_err_hist: np.array of shape (hist_len,) -> 历史小时电价预测误差数据
    - N_S: integer -> 要生成的场景数
    - max_wind: float -> 最大风速安全阈值
    - max_wave: floar -> 最大浪高安全阈值
    - rle_mean: np.array of shape (N_t,) -> 每个风力涡轮机的剩余寿命估计平均值
    - rle_std: float or np.array of floats of shape (N_t,) -> RLE预测STD用于随机场景发生器
    - tau: integer np.array of shape (N_t,) -> 完成每项维护任务所需的工时
    - rle_dist: (optional) str -> 被用来生成场景的分布，有效的输入是正态分布和威布尔分布
        如果选择weibull，则使用rle_std作为形状参数，rle_mean作为尺度参数。
    - tR: (optional) integer -> 以24小时为单位的首次日照时间(默认=5)
    - tD: (optional) integer -> 以24小时为单位的最后一次日照时间(默认为21) 
    - hist_len (optional) integer -> 场景生成的自定义历史记录长度(以天为单位)(默认=5)
    - random_seed: (optional) integer -> 用于场景生成再现性的种子(默认=1)
    - sim_day_ahead: (optional) bool -> 使用预定义的时间表和真实数据/无场景模拟前一天 
        (default=False)
        
    Returns:
    - norm_power: np.array of shape (24, N_t, N_S) -> STH内风力发电机的每小时标准化功率情景
    - norm_powerL: np.array of shape (dJ, N_t, S) -> LTH风电机组的日归一化功率情景
    - el_price: np.array of shape (24, N_S) -> 运输及房屋局的每小时电价情景[元/兆瓦时]
    - el_priceL: np.array of shape (dJ, N_S) -> LTH日电价情景[$/MWh]
    - op_time: np.array of shape (24, N_t, N_S) -> 涡轮机维修每小时运行时间方案[小时]
    - op_timeL: np.array of shape (dJ, N_t, N_S) -> 汽轮机维护日运行时间情景[小时]
    - rle: np.array of shape (N_t, N_S) -> RLE场景
    - access: np.array of shape (24, N_t, N_S) -> STH的可访问性
    - access_LTH: np.array of shape (dJ, N_t, N_S) -> LTH的可访问性
    - ws_STH_scenarios: np.array of shape (24, N_t, N_S) -> STH 风速场景
    - wh_STH_scenarios: np.array of shape (24, N_S) -> STH 浪高场景
    - ws_LTH_scenarios: np.array of shape (dJ, N_t, N_S) -> LTH 风速场景 
    - wh_LTH_scenarios: np.array of shape (dJ, N_S) -> LTH 浪高场景
    """
    dJ, N_t = ws_LTH.shape
    
    t1=time()
    
    if (sim_day_ahead) | (N_S==1):
        ws_STH_scenarios = ws_STH[:,:,np.newaxis]
        wh_STH_scenarios = wh_STH[:,np.newaxis]
        ws_LTH_scenarios = ws_LTH[:,:,np.newaxis]
        wh_LTH_scenarios = wh_LTH[:,np.newaxis]
        
        el_price = ep_STH[:,np.newaxis]
        el_priceL = ep_LTH[:,np.newaxis]

    else:
        ws_STH_scenarios = np.repeat(ws_STH[:,:,np.newaxis], N_S, -1,) 
        ws_LTH_scenarios = np.repeat(ws_LTH[:,:,np.newaxis], N_S, -1,) 
        
        wh_STH_scenarios = np.repeat(wh_STH[:,np.newaxis], N_S, -1,) 
        wh_LTH_scenarios = np.repeat(wh_LTH[:,np.newaxis], N_S, -1,) 
        
        el_price = np.repeat(ep_STH[:,np.newaxis], N_S, -1,) 
        el_priceL = np.repeat( ep_LTH[:,np.newaxis], N_S, -1,) 
        
        if NAIVE:
            np.random.seed(random_seed)
            for i in range(N_t):
                ws_STH_scenarios[:,i,:] -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),ws_err_hist[:,i].std()),size=(24,N_S))
                ws_LTH_scenarios[:,i,:] -= np.random.normal(loc=np.zeros((dJ,1)),
                                scale=np.full((dJ,1),ws_err_hist[:,i].std()),size=(dJ,N_S))
            
                wh_STH_scenarios -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),wh_err_hist.std()),size=(24,N_S))
                wh_LTH_scenarios -= np.random.normal(loc=np.zeros((dJ,1)),
                                scale=np.full((dJ,1),wh_err_hist.std()),size=(dJ,N_S))
                
                el_price -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),ep_err_hist.std()),size=(24,N_S))
                el_priceL -= np.random.normal(loc=np.zeros((dJ,1)),
                                scale=np.full((dJ,1),ep_err_hist.std()),size=(dJ,N_S))
        else:
            for i in range(N_t):
                if not(BESN): ws_STH_scenarios[:,i,:] -= error_scenario_gen(ws_err_hist[:,i], 24, N_S, 
                      24*hist_len, random_seed, DGP=DGP,GPY=GPY,MAT=MAT,GS=GS,LSTM=LSTM,MLP=MLP)
                ws_LTH_scenarios[:,i,:] -= error_scenario_gen(ws_err_hist[:,i].reshape(-1,24).mean(1),
                                          dJ+1, N_S, 50, random_seed, LDS=True, DGP=DGP,GPY=GPY,MAT=MAT,GS=GS,LSTM=LSTM,MLP=MLP)[1:,:] #discard the day-ahead
        
            if not(BESN): wh_STH_scenarios -= error_scenario_gen(wh_err_hist, 24, N_S, 24*hist_len, 
                  random_seed, LDS=True, DGP=DGP,GPY=GPY,MAT=MAT,GS=GS,LSTM=LSTM,MLP=MLP)
            wh_LTH_scenarios -= error_scenario_gen(wh_err_hist.reshape(-1,24).mean(1),
                                          dJ+1, N_S, 50, random_seed, LDS=True, DGP=DGP,GPY=GPY,MAT=MAT,GS=GS,LSTM=LSTM,MLP=MLP)[1:,:] #discard the day-ahead
            
            el_price -= error_scenario_gen(ep_err_hist, 24, N_S, 24*hist_len, random_seed, DGP=DGP,GPY=GPY,MAT=MAT,GS=GS,LSTM=LSTM,MLP=MLP)
            el_priceL -= error_scenario_gen(ep_err_hist.reshape(-1,24).mean(1),
                                          dJ+1, N_S, 50, random_seed, DGP=DGP,GPY=GPY,MAT=MAT,GS=GS,LSTM=LSTM,MLP=MLP)[1:,:] #discard the day-ahead
        
        ws_STH_scenarios[ws_STH_scenarios<0.5]=0.5
        wh_STH_scenarios[wh_STH_scenarios<0.1]=0.1
        ws_LTH_scenarios[ws_LTH_scenarios<0.5]=0.5
        wh_LTH_scenarios[wh_LTH_scenarios<0.1]=0.1
        

    norm_power = binning_method(ws_STH_scenarios)
    norm_powerL = binning_method(ws_LTH_scenarios)
    
    op_time, access = get_mission_time_STH(
            ws_STH_scenarios,
            wh_STH_scenarios,
            max_wave,
            max_wind,
            tau,
            tR,
            tD)
    

    op_timeL, access_LTH = get_mission_time_LTH(ws_LTH_scenarios, 
                                                wh_LTH_scenarios, 
                                                tau)
        
    if (sim_day_ahead) | (N_S==1) & (not (DGP)):
        rle = rle_mean[:,np.newaxis]
    else:
        if DGP:
            rle = (rle_mean*gamma(1+1/rle_std)).reshape(-1,1)
        else:
            rle = rle_scenario_gen(rle_mean, rle_std, N_S, rle_dist, random_seed)
        
    print('Data prepared in ', round(time()-t1, 4), ' sec')
    
    return norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, rle, access, access_LTH, ws_STH_scenarios, wh_STH_scenarios, ws_LTH_scenarios, wh_LTH_scenarios


# In[10]:


def get_var_from_model(model_dict, var_name, STH_len=24, LTH_len=19, N_t=5):
    """
    函数用于将存储为字典项的模型变量提取为numpy数组。
    
    Inputs:
    - model_dict: 一个字典，其键是以字符串格式表示的模型的单变量名称（'var_name [time_idx，wt_idx，scenario_idx]'），存储相应的变量值。
    - var_name: 一个带有变量块名称的字符串。 
    - STH_len: (optional; Default=24) The length of the STH.
    - LTH_len: (optional; Default=19) The length of the LTH.
    - N_t: (optional; Default=5) The number of wind turbines considered.
    
    Returns:
    - var_value_array: 一个包含查询变量值的numpy数组。
    - var_name_array: 一个包含每个单独变量名称的numpy数组。
    """
    var_name_list = []
    var_value_list = []
    for var, value in model_dict.items():
        if var_name+'[' in var:
            var_name_list.append(var)
            var_value_list.append(value)
    
    var_name_array = np.array(var_name_list)
    var_value_array = np.array(var_value_list)
    
    if '[d' in var_name_array[0]:
        if 'wt' in var_name_array[0]:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(LTH_len,N_t,-1)
                var_value_array = var_value_array.reshape(LTH_len,N_t,-1)
            else:
                var_name_array = var_name_array.reshape(LTH_len,N_t)
                var_value_array = var_value_array.reshape(LTH_len,N_t)
        else:
            if ',s]' in var_name_array[0]:
                var_name_array = var_name_array.reshape(LTH_len,-1)
                var_value_array = var_value_array.reshape(LTH_len,-1)
            else:
                var_name_array = var_name_array.reshape(LTH_len)
                var_value_array = var_value_array.reshape(LTH_len)
                
    elif '[t' in var_name_array[0]:
        if 'wt' in var_name_array[0]:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(STH_len,N_t,-1)
                var_value_array = var_value_array.reshape(STH_len,N_t,-1)
            else:
                var_name_array = var_name_array.reshape(STH_len,N_t)
                var_value_array = var_value_array.reshape(STH_len,N_t)
        else:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(STH_len,-1)
                var_value_array = var_value_array.reshape(STH_len,-1)
            else:
                var_name_array = var_name_array.reshape(STH_len)
                var_value_array = var_value_array.reshape(STH_len)
    
    elif '[wt' in var_name_array[0]:
        var_name_array = var_name_array.reshape(N_t,-1)
        var_value_array = var_value_array.reshape(N_t,-1)
    else:
        var_name_array = var_name_array.reshape(-1,1)
        var_value_array = var_value_array.reshape(-1,1)
    
    if var_name=='v':
        return np.array(model_dict['v']), np.array(['v'])
    
    else:
        return var_value_array, var_name_array


# In[14]:


def RH_iter_solver( wind_speed_true,
                    wave_height_true,
                    wind_speed_forecast,
                    wave_height_forecast,
                    N_S,
                    Din,
                    max_wind, 
                    max_wave,
                    el_prices_true,
                    el_prices_forecast,
                    rle_true,
                    rle_forecast_mean,
                    rle_forecast_std,
                    tau,
                    rho,
                    theta,
                    rle_dist='weibull',
                    tR=5, 
                    tD=21,
                    dJ=19,
                    hist_len=5,
                    random_seed=1,
                    max_iter=None,
                    mip_gap=0.001,
                    CMS=False,
                    AIMS=False,
                    BESN=False,
                    TBS=False,
                    NAIVE=False,
                    DGP=False,
                    GPY=False,
                    MAT=False,
                    GS=False,
                    LSTM=False,
                    MLP=False):
    """
    使用滚动视野迭代算法的函数，将随机模型的输出作为输入，使用真实数据的模拟模型获取真实的进度成本。
    
    Inputs:
    -wind_speed_true: np.array of shape (obs_dim, N_t) -> 将在日前模拟中使用的真实风速数据。
    -wave_height_true: np.array of shape (obs_dim, N_t) -> 将在日前模拟中使用的真实波高数据。
    -wind_speed_forecast: np.array of shape (obs_dim, N_t) -> 将被用于随机程序的预测风速数据。
    -wave_height_forecast: np.array of shape (obs_dim, N_t) -> 将用于随机程序的预测波高数据。
    -N_S: int -> 生成的场景数量
    -Din: int -> 数据集的第一天是哪一天
    -wind_scenario_std: float or np.array of (time_dim, N_t) -> 用于情景生成的风速点预报的STH。
    -wave_scenario_std: : float or np.array of (time_dim, N_t) -> STH的波高点预测将用于情景生成。
    -max_wind: float -> 船舶/船员操作的最大风速容忍度
    -max_wave: float -> 船舶/船员操作的最大波高容忍度 
    -el_prices_true: np.array of shape (time_dim, ) -> 将在日前模拟中使用的真实电价配置文件。
    -el_prices_forecast: np.array of shape (time_dim, ) -> 将用于随机程序的预测电价曲线。
    -rle_true: np.array of shape (N_t, ) -> 将用于日前（以天为单位）模拟的真实RLE。
    -rle_forecast: np.array of shape (N_t, ) -> 将在随机程序中使用的预测的RLE（以天为单位）。
    -rle_std: np.array of shape (N_t, ) -> 将在随机程序中使用的RLE STD（以天为单位）
    -tau: int np.array of shape (N_t, ) -> 完成每项任务所需的时间
    -rho: bool np.array of shape (N_t, ) -> 先前已启动维护
    -theta: bool np.array of shape (N_t, ) -> 需要维护
    
    
    Optional:
    - rle_dist: (default='weibull') str -> 用于场景生成的分布。有效输入为“正态”和“威布尔”。如果选择“威布尔”，则rle_std用作形状参数，rle_mean用作位置参数。
    - tR: int (default=5) -> 最早工作开始时间 
    - tD: int (default=21) -> 最晚工作结束时间
    - dJ: int (default=19) -> 长期优化范围天数
    - hist_len: int (default=5) -> 用于情景生成的GPR的历史长度，以天为单位。
    - random_seed: int (default=1) -> 随机数种子
    - max_iter: int (default=None) -> 最大迭代次数。如果选择了None，则地平线将滚动，直到所有风力涡轮完成维护任务。
    - mip_gap: float (default=0.001) -> 随机解的最大相对差优化
    
    Returns:
    -output: 一个带有条目的字典：
    'stochastic_sol': 一个列表，其项是每个时间段滚动时STOCHOS的字典输出。
    'stochastic_input': 一个元组，其元素是STOCHOS每个时间段滚动的随机输入的列表。每个列表具有以下顺序的输入：
            0. norm_power
            1. norm_powerL
            2. el_price
            3. el_priceL
            4. op_time
            5. op_timeL
            6. access
            7. ws_STH
            8. wh_STH
            9. rle
            10. theta
            11. rho
    'simulation': 一个列表，其中每个项都是在使用STH计划的情况下，模拟每个视野滚动时'stochastic_sol'的字典输出。
    'true_input': 一个元组，其元素是模拟模型在每个时间段滚动后的真实输入列表，按照与“随机输入”相同的顺序。
    'time_per_roll': 每个视野滚动的优化时间列表
    'total_hourly_sched_time': 所有卷的总优化时间的浮点数
    """
    
    N_t = wind_speed_true.shape[1]
    
    stochastic_input = []
    stochastic_output = []
    
    true_input = []
    true_output = []
    one_iteration_time = []
    
    rle_true = rle_true.copy()
    rle_forecast_mean = rle_forecast_mean.copy()
    tau = tau.copy()
    rho = rho.copy()
    theta = theta.copy()
    
    t0 = time()

    run=0
    while np.any(theta>0) | np.any(rho>0):
        t1 = time()
        
        wind_speed_error_hist = wind_speed_forecast[:(Din+run)*24,:]-wind_speed_true[:(Din+run)*24,:]
        wave_height_error_hist = wave_height_forecast[:(Din+run)*24]-wave_height_true[:(Din+run)*24]
        el_price_error_hist = el_prices_forecast[:(Din+run)*24]-el_prices_true[:(Din+run)*24]
        
        wsf_STH = wind_speed_forecast[24*(Din+run):24*(Din+run+1),:]
        whf_STH = wave_height_forecast[24*(Din+run):24*(Din+run+1)]
        wsf_LTH = wind_speed_forecast[24*(Din+run+1):24*(Din+run+dJ+1),:].reshape(-1,24,N_t).mean(1)
        whf_LTH = wave_height_forecast[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        
        wst_STH = wind_speed_true[24*(Din+run):24*(Din+run+1),:]
        wht_STH = wave_height_true[24*(Din+run):24*(Din+run+1)]
        wst_LTH = wind_speed_true[24*(Din+run+1):24*(Din+run+dJ+1),:].reshape(-1,24,N_t).mean(1)
        wht_LTH = wave_height_true[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        
        epf_STH = el_prices_forecast[24*(Din+run):24*(Din+run+1)]
        ept_STH = el_prices_true[24*(Din+run):24*(Din+run+1)]
        epf_LTH = el_prices_forecast[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        ept_LTH = el_prices_true[24*(Din+run+1):24*(Din+run+dJ+1)].reshape(-1,24).mean(1)
        

        (norm_power, 
         norm_powerL, 
         el_price, 
         el_priceL, 
         op_time, 
         op_timeL, 
         rle,
         access, access_LTH ,ws_STH,wh_STH,_,_) = data_loader(
                 wsf_STH, 
                 wsf_LTH, 
                 whf_STH,
                 whf_LTH,
                 epf_STH,
                 epf_LTH,
                 wind_speed_error_hist,
                 wave_height_error_hist,
                 el_price_error_hist,
                 N_S, 
                 max_wind, 
                 max_wave,
                 rle_forecast_mean,
                 rle_forecast_std,
                 tau,
                 rle_dist,
                 tR, 
                 tD,
                 hist_len,
                 random_seed,
                 sim_day_ahead=False,
                 BESN=BESN,
                 NAIVE=NAIVE,
                 DGP=DGP,
                 GPY=GPY,
                 MAT=MAT,
                 GS=GS,
                 LSTM=LSTM,
                 MLP=MLP)
        
        # (norm_power, 
        #  norm_powerL, 
        #  el_price, 
        #  el_priceL, 
        #  op_time, 
        #  op_timeL, 
        #  rle,
        #  access, access_LTH,ws_STH,wh_STH,_,_) = data_loader(
        #          wst_STH, 
        #          wst_LTH, 
        #          wht_STH,
        #          wht_LTH,
        #          ept_STH,
        #          ept_LTH,
        #          wind_speed_error_hist,
        #          wave_height_error_hist,
        #          el_price_error_hist,
        #          N_S, 
        #          max_wind, 
        #          max_wave,
        #          rle_true,
        #          0,
        #          tau,
        #          rle_dist,
        #          tR, 
        #          tD,
        #          hist_len,
        #          random_seed,
        #          sim_day_ahead=True,
        #          GPY=GPY,
        #          MAT=MAT,
        #          GS=GS,
        #          LSTM=LSTM)
        
        if AIMS:
            op_time = np.repeat(np.repeat(tau[np.newaxis,:,np.newaxis],24,0),N_S,2)
            op_timeL = np.repeat(np.repeat(tau[np.newaxis,:,np.newaxis],dJ,0),N_S,2)
            access_LTH = np.ones((dJ,N_t,N_S))
        
        if BESN:
            #el_price = np.ones((24,N_S))*60*1.15 #Besnard 2011 electricity price = 60 euros/MWh
            #el_priceL = np.ones((dJ,N_S))*60*1.15 #Besnard 2011 electricity price = 60 euros/MWh
            #el_price = np.repeat(epf_STH.copy()[:,np.newaxis],N_S,1)
            #el_priceL = np.repeat(epf_LTH.copy()[:,np.newaxis],N_S,1)
            el_price = np.ones((24,N_S))*np.mean(el_prices_forecast) 
            el_priceL = np.ones((dJ,N_S))*np.mean(el_prices_forecast) 
            rle = np.repeat(rle_forecast_mean[:,np.newaxis],N_S,1)
        
        stochastic_input.append((norm_power.copy(),norm_powerL.copy(),el_price.copy(),el_priceL.copy(),op_time.copy(),
                                 op_timeL.copy(),access.copy(), ws_STH.copy(), wh_STH.copy(), access_LTH.copy(),
                                 rle.copy(),theta.copy(),rho.copy()))
        
        stochastic_output.append(STOCHOS(norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, rle, access_LTH, theta,
                                         rho, curtailment=0, scalars=None, STH_sched=None, return_all_vars=True,
                                         rel_gap = mip_gap, CMS=CMS, TBS=TBS))
        
        if stochastic_output[-1]==None: break
        
        stoch_STH_sched = stochastic_output[run]['STH_sched']

        (norm_power, 
         norm_powerL, 
         el_price, 
         el_priceL, 
         op_time, 
         op_timeL, 
         rle,
         access, access_LTH,ws_STH,wh_STH,_,_) = data_loader(
                 wst_STH, 
                 wst_LTH, 
                 wht_STH,
                 wht_LTH,
                 ept_STH,
                 ept_LTH,
                 wind_speed_error_hist,
                 wave_height_error_hist,
                 el_price_error_hist,
                 N_S, 
                 max_wind, 
                 max_wave,
                 rle_true,
                 0,
                 tau,
                 rle_dist,
                 tR, 
                 tD,
                 hist_len,
                 random_seed,
                 sim_day_ahead=True,
                 GPY=GPY,
                 MAT=MAT,
                 GS=GS,
                 LSTM=LSTM,
                 MLP=MLP)
        
        true_input.append((norm_power.copy(),norm_powerL.copy(),el_price.copy(),el_priceL.copy(),op_time.copy(),
                           op_timeL.copy(),access.copy(), ws_STH.copy(), wh_STH.copy(), access_LTH.copy(),
                           rle.copy(),theta.copy(),rho.copy()))
        
        true_output.append(STOCHOS(norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, rle, access_LTH, theta,
                                   rho, curtailment=0, scalars=None, STH_sched=stoch_STH_sched, return_all_vars=True,
                                   rel_gap = 0.0))
        
        if true_output[-1]==None: break
        
        #随着我们向前移动一天，将RLE平均值减少1天：
        rle_true -= 1
        rle_true[rle_true<0]=0
        
        rle_forecast_mean -= 1
        rle_forecast_mean[rle_forecast_mean<0]=0
        
        #如果真实的RLE为零，则将预测的平均值设置为零。
        rle_forecast_mean[rle_true==0] = 0
        
        #如果预测的RLE平均值为0天，但是风力涡轮仍在运转，则通过将其设置为1天来更正预测平均值：
        rle_forecast_mean[(rle_forecast_mean==0) & (rle_true>0)] = 1
        
        
        
        true_STH = np.array(true_output[run]['STH_sched'])
        
        wind_speed_error_hist = np.append(wind_speed_error_hist,
                                          wsf_STH-wst_STH, 0)
        wave_height_error_hist = np.append(wave_height_error_hist,
                                          whf_STH-wht_STH, 0)
        
        
        #检查未完成的任务或在STH中已完成的任务：
        for i in range(N_t):
            if (np.any(true_STH[:,i])>0):
                if true_output[run]['remaining_maint_hrs'][i] == 0:
                    theta[i]=0
                    rho[i]=0
                else:
                    theta[i]=1
                    rho[i]=1
                    tau[i]=true_output[run]['remaining_maint_hrs'][i,0]

        print('theta:', theta)
        print('rho:', rho)
        print('tau:', tau)

        run+=1
        one_iteration_time.append(time()-t1)
        
        if run==max_iter: break
        
    return {  'stochastic_sol':stochastic_output,
              'stochastic_input': stochastic_input,
              'simulation': true_output,
              'true_input':true_input,
              'time_per_roll': one_iteration_time, #一次迭代的时间
              'total_hourly_sched_time': time()-t0 }


# In[23]:


def show_full_schedule(solution, return_schedule = False):
    """
    生成所有运行的STH时间表的甘特图。
    真实的残留寿命用橙色竖线显示。
    RLE预测方案以红色垂直线显示。
    如果return_schedule设置为true，则还会返回完整的小时计划表。
    """    
    schedule = np.array(solution['simulation'][0]['STH_sched'])
    op_time = solution['true_input'][0][4]

    for i in range(1,len(solution['time_per_roll'])):
        schedule = np.append(schedule,np.array(solution['simulation'][i]['STH_sched']), 0)
        op_time = np.append(op_time, solution['true_input'][i][4],0)

    rle_true = solution['true_input'][0][-3]
    rle_stoch = solution['stochastic_input'][0][-3]

    
    N_t, N_S = rle_stoch.shape
    
    trns = 1 if N_S==1 else 0.3

    x = np.arange(schedule.shape[0])#.reshape(-1,1)#, N_t, 1)
    rle_true = 24*rle_true.copy()
    rle_stoch = 24*rle_stoch.copy()
    
    #print('schedule.shape',schedule.shape)
    #('x.shape',x.shape)
    #print('op_time.shape',op_time.shape)
    
    for i in range(N_t):
        plt.vlines(rle_true[i],i+1-0.5,i+1+0.5, color='orange', linewidth=3,alpha=1)
        plt.hlines(np.full(schedule.shape[0], i+1),x*schedule[:,i],(x+op_time[:,i,0])*schedule[:,i], 
                           color='black', linewidth=15, alpha=1)
        plt.vlines(rle_stoch[i,:],np.full(N_S, i+1-0.5),np.full(N_S, i+1+0.5), color='red', linewidth=2,alpha=trns)

    if return_schedule: return schedule


#%%
    
def show_full_revenue(solution, return_revenue=False, patch_LTH_revenue=False, R=12):
    """
    返回预期收入的完整每小时收入概况，累加所有风力涡轮机的收入。
    真实收入以红线显示，方案以灰线显示。
    R: float -> 风力涡轮机的额定功率。
    如果return_revenue设置为True，它还返回一个包含实际小时收入和收入的对
预测场景为np数组。
    如果patch_LTH_revenue被设置为True，它还返回上一次滚动的LTH收入，
有了每天的方案
    """
    true_revenue =  np.array(solution['true_input'][0][2]*solution['true_input'][0][0].sum(1)*R)
    stoch_revenue = np.array(solution['stochastic_input'][0][2]*solution['stochastic_input'][0][0].sum(1)*R)
    
    dJ, N_S = stoch_revenue.shape
    
    
    
    for i in range(1,len(solution['time_per_roll'])):
        true_revenue = np.append(true_revenue,np.array(
                solution['true_input'][i][2]*solution['true_input'][i][0].sum(1)*R), 0)
        stoch_revenue = np.append(stoch_revenue,np.array(
                solution['stochastic_input'][i][2]*solution['stochastic_input'][i][0].sum(1)*R), 0)
    
    if patch_LTH_revenue:
        true_LTH_rev = np.repeat((solution['true_input'][-1][3]*solution['true_input'][-1][1].sum(1)*R
                                  )[:,:,np.newaxis],24,-1).reshape(-1,1)
        stoch_LTH_rev = np.repeat((solution['stochastic_input'][-1][3]*solution['stochastic_input'][-1][1].sum(1)*R
                                   )[:,:,np.newaxis],24,-1).transpose(0,2,1).reshape(-1,N_S)
        
        true_revenue = np.append(true_revenue, true_LTH_rev, 0)
        stoch_revenue = np.append(stoch_revenue, stoch_LTH_rev, 0)
    
    
    plt.plot(true_revenue[:,0], color='red', linewidth=2)
    plt.plot(stoch_revenue, color='grey', alpha=0.5)
    plt.plot(true_revenue[:,0], color='red', linewidth=2)
    plt.ylabel('Total revenue ($)')
    
    if return_revenue: return (true_revenue, stoch_revenue)


# In[24]:


def show_LTH_schedule(solution, roll=0 , return_schedule=False):
    """
    生成一个申请表格，用于展示在一个卷中获得的LTH日程安排。 
    真实残余寿命以橙色竖线显示。
    RLE预测方案显示为红色垂直线。
    如果return_schedule设置为true，则还会返回此卷的LTH计划。
    """    
    x = solution['stochastic_sol'][roll]['LTH_sched'].copy()
    op_time = solution['stochastic_input'][roll][5]
    rle = solution['stochastic_input'][roll][-3]
    rle_true = solution['true_input'][roll][-3]
    
    dJ, N_t, N_S = op_time.shape
        
    for i in range(N_t):
        x[:,i,:] = x[:,i,:]*(i+1)
        plt.vlines(rle[i,:]-1,np.full(N_S,i+1-0.5),np.full(N_S,i+1+0.5), color='red', linewidth=1,alpha=0.2)
        plt.vlines(rle[i,:].mean()-1,i+1-0.5,i+1+0.5, color='red', linewidth=2,alpha=1)
        plt.vlines(rle_true[i]-1,i+1-0.5,i+1+0.5, color='orange', linewidth=2,alpha=1)
        for j in range(dJ):
            plt.hlines(np.full(N_S,i+1),j*x[j,i,:]/(i+1),j*x[j,i,:]/(i+1)+x[j,i,:]*op_time[j,i,:]/24, 
                       color='black', linewidth=15, alpha=0.1)
    plt.ylim([0.5,N_t+0.5])
    plt.yticks([i for i in range(1,N_t+1)], ['WT'+str(i) for i in range(1,N_t+1)])
    plt.xticks([i for i in range(dJ)],[i for i in range(1,dJ+1)])
    plt.xlabel('Day of the LTH')
            
    if return_schedule: return x



# In[25]:

def show_weather_profile(wind_speed_true, 
                         wave_height_true,
                         wind_speed_forecast, 
                         wave_height_forecast,
                         Din, 
                         N_S,
                         turbine = 0,
                         horizon_len=5,
                         max_wind=15, 
                         max_wave=1.5):
    
    N_t = wind_speed_true.shape[1]
    ws_scenarios=np.zeros((24*horizon_len,N_t,N_S))
    wh_scenarios=np.zeros((24*horizon_len,N_S))
    access = np.zeros((24*horizon_len, N_t, N_S))
    
    for run in range(horizon_len):
        wind_speed_error_hist = wind_speed_forecast[:(Din+run)*24,:]-wind_speed_true[:(Din+run)*24,:]
        wave_height_error_hist = wave_height_forecast[:(Din+run)*24]-wave_height_true[:(Din+run)*24]
        wsf_STH = wind_speed_forecast[24*(Din+run):24*(Din+run+1),:]
        whf_STH = wave_height_forecast[24*(Din+run):24*(Din+run+1)]
            
        _, _, _, _, _, _, _, acc, wss, whs, _, _ = data_loader( ws_STH=wsf_STH,  
                                                                ws_LTH=np.zeros((1,N_t)),
                                                                wh_STH=whf_STH,
                                                                wh_LTH=np.zeros((1,)),
                                                                ep_STH=np.zeros((24,)),
                                                                ep_LTH=np.zeros((1,)), 
                                                                ws_err_hist=wind_speed_error_hist,
                                                                wh_err_hist=wave_height_error_hist,
                                                                ep_err_hist=np.zeros(100*24),
                                                                N_S=N_S,
                                                                max_wind=max_wind, 
                                                                max_wave=max_wave,
                                                                rle_mean=np.ones(N_t),
                                                                rle_std=np.zeros(N_t),
                                                                tau=np.ones(N_t),
                                                                rle_dist='weibull',
                                                                tR=5, 
                                                                tD=21,
                                                                random_seed=1,
                                                                sim_day_ahead=False
                                                               )
    
        ws_scenarios[24*run:24*(run+1),:,:] = wss.copy()
        wh_scenarios[24*run:24*(run+1),:] = whs.copy()
        access[24*run:24*(run+1),:,:] = acc.copy()
        
    plt.plot(ws_scenarios[:,turbine,:], color='grey', alpha=0.25)
    plt.plot(wind_speed_forecast[Din*24:(Din+horizon_len)*24,turbine], color='grey', linewidth=2)
    plt.plot(wh_scenarios[:,:], color=[0.7, 0.8, 1], alpha=0.25)
    plt.plot(wave_height_forecast[Din*24:(Din+horizon_len)*24], color='cyan', linewidth=2)
    plt.plot(wind_speed_true[Din*24:(Din+horizon_len)*24,turbine], color='black',linewidth=2)
    plt.plot(wave_height_true[Din*24:(Din+horizon_len)*24], color='blue',linewidth=2)
    plt.hlines(max_wind, 0, 24*horizon_len, color='black', linestyle='--')
    plt.hlines(max_wave, 0, 24*horizon_len, color='blue', linestyle='--')
    
    return access
        

def make_light(sol_dict):
    for i in sol_dict.keys():
        for j in range(len(sol_dict[i]['time_per_roll'])):
            sol_dict[i]['stochastic_sol'][j]['model'] = None
            sol_dict[i]['simulation'][j]['model'] = None
    return sol_dict


