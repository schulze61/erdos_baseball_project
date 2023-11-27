
# Collection of codes used to predict a baseball game

import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma, factorial
import statsmodels.api as sm


#===============================================Reading Data=================================================
def data_read(year_now):
    """
    reads data of the given year
    """
    
    path = './game_log_sorter/new_sorted_game_logs/gl' + str(year_now) + '_new_sorted.csv'
    data = pd.read_csv(path, header=0, index_col=0)

    return data



#====================================================team_season============================================
def team_data(team, season_data, pe_exp):
    """
    extract the entire season of any "team" from the "season_date", given "pe_exp", exponent to compute PE
    returns a sorted dataframe for that specific team
    """
    
    team_index = season_data[(season_data["team1"] == team) | (season_data["team2"] == team)].index
    team_data  = pd.DataFrame(index = range(len(team_index)))
    
    for ind_team, ind in zip(team_data.index, team_index):
    
        team_data.loc[ind_team, "date"] = season_data.loc[ind, "date"]

        if (season_data.loc[ind, "team1"] == team):

            team_data.loc[ind_team, "g_n"] = season_data.loc[ind, "team1_gn"]
            team_data.loc[ind_team, "r_s"] = season_data.loc[ind, "team1_rs"]
            team_data.loc[ind_team, "r_a"] = season_data.loc[ind, "team1_ra"]
        else:       

            team_data.loc[ind_team, "g_n"] = season_data.loc[ind, "team2_gn"]
            team_data.loc[ind_team, "r_s"] = season_data.loc[ind, "team2_rs"]
            team_data.loc[ind_team, "r_a"] = season_data.loc[ind, "team2_ra"]

    
    # sort dataframe based on game number
    team_data = team_data.sort_values('g_n').copy() 
    team_data.reset_index(inplace = True)

#     cumulative runs scored and allowed
    team_data["cum_rs"] = team_data["r_s"].cumsum()
    team_data["cum_ra"] = team_data["r_a"].cumsum()
    team_data["pe"]     = team_data["cum_rs"]**pe_exp /(team_data["cum_rs"]**pe_exp + team_data["cum_ra"]**pe_exp)
    
    
    team_data["dpe"]    = np.abs(team_data["pe"] - team_data["pe"].shift(+1))
    team_data.loc[team_data[team_data["dpe"]<1.0e-5].index, "dpe"] = 1.0e-5   
    return team_data  


#===============================================parameters of simulation from data=================================================

def params(season_data, pe_exp, shape, rs_0, method = 1):
    """
    1) runs the regression of dpe ~ game_n without (method = 1) or with (method = 2) an intercept
    2) gives the beysian updating scale parameter for each week
    """
    
    teams = list(season_data.team1.unique()) + list(set(season_data.team2.unique()) - set(season_data.team1.unique()))
    teams = sorted(teams).copy()
    
    Result  = pd.DataFrame(index = teams)
    scale_b = pd.DataFrame(index = range(season_data["team1_gn"].max()), columns = teams)
    
    # extract entire season for a team
    for team in teams:
        # dataframe for this team's season
        data_now = team_data(team, season_data, pe_exp)

        # Regression for tolerance parameters: I drop the first game 
        y = np.log(data_now.loc[1:, "dpe"])
        
        if method == 1:
            x     = np.log(data_now.loc[1:, "g_n"])
            model = sm.OLS(y, x).fit()
            Result.loc[team , "eps_1"] = 1.0 
            Result.loc[team , "eps_2"] = -1.0 * np.round(model.params[0], 4)
            
            
        elif method == 2:
            x     = sm.add_constant(np.log(data_now.loc[1:, "g_n"]))
            model = sm.OLS(y, x).fit()
            
            Result.loc[team , "eps_1"] = np.round(np.exp(model.params[0]), 4)
            Result.loc[team , "eps_2"] = -1.0 * np.round(model.params[1], 4)
        

        Result.loc[team , "R2"]      = np.round(model.rsquared, 4)
        
        # Bayesian Updating of Scale
        scale_b[team] = bayes_scale_update(data_now["r_s"], shape, rs_0)

#         plt.plot(data_now["g_n"], data_now["delta_pe"])

    return Result, scale_b



#====================================================Scale_param-with-Bayes============================================

def bayes_scale_update(rs_series, shape, rs_0):
    """
    For a given "team", uses runs scored in each game to update scale param. of the weibull distribution
    """
    
    # initial assumption: each team scores 4 runs [league average 4-5]
    xbar = (rs_0/math.gamma(1+(1.0/shape)))**shape
    b_0  = xbar
    a_0  = 2.0

    # updated scale parameter at the end of week n
    rs_k  = rs_series**shape
    b_sum = rs_k.cumsum()

    scale_bayes = pd.DataFrame(index = rs_series.index, columns = ["Bayes_Scale"])
    scale_bayes.loc[0, "Bayes_Scale"] = b_0**(1.0/shape)

    for game_n in range(1, len(scale_bayes.index)):

        a_now = a_0 + game_n - 1.0
        b_now = b_0 + b_sum[game_n - 1]

        scale_bayes.loc[game_n, "Bayes_Scale"] = (b_now/a_now)**(1.0/shape)

    return scale_bayes



#===============================================pythagorean expectations=================================================
def PE(rs, ra, n_g, method = "basic"):
    """
    Computes PE given inputs "runs scored", "runs allowed", "number of games", and a specified method
    
    methods include: {basic, davenport, smyth}
    """
    if (rs == 0) & (ra == 0):
        
        pe_out = 0.5
        
    elif method == "basic":
        
        c = 1.83
        pe_out = (rs**c)/(rs**c + ra**c)
      
    elif method == "davenport":
        
        c =  1.50 * np.log10((rs+ra)/n_g) + 0.45     # is it log or log10? 
        pe_out = (rs**c)/(rs**c + ra**c)
        
    elif method == "smyth":    
        
        c = ((rs+ra)/n_g)**0.287
        pe_out = (rs**c)/(rs**c + ra**c)
        
    else:
        print("method not valid!")
    
    return pe_out


#====================================================Scale Parameter============================================
def scale_param(rs_avg, shape, beta = -0.5):
    """
    computes the scale parameter for RS, given the shape parameter
    """
    
    return (rs_avg - beta)/gamma(1 + 1.0/shape)

#====================================================draw_weibull============================================
def draw_weibull(scale, shape):
    """
    returns a rv from a two-parmeter Weibull
    """
    return scale * np.random.weibull(shape)



#====================================================weibull_pdf============================================
def weibull_pdf(x, scale, shape):
    """
    Returns PDF of a given value x for a two-parmeter Weibull
    """
    if x >= 0:
        val = (shape/scale) * ((x/scale)**(shape-1.0)) * np.exp(-((x/scale)**shape))
    else:
        val = 0.0
    
    return val



#====================================================weibull_cdf============================================
def weibull_cdf(x, scale, shape):
    """
    Returns PDF of a given value x for a two-parmeter Weibull
    """
    if x >= 0:
        val = 1.0 - np.exp(-1.0 * (x/scale)**(shape))
    else:
        val = 0.0
    
    return val


#====================================================simulation_cdf============================================
def runs_scored_weibull(scale, shape):
    """"
    draws a random number from a weibull dist. for runs scored
    returns: 1) runs scored, 2) prob. of draw
    """    
    rs_draw = draw_weibull(scale, shape)
    pr_draw = weibull_pdf(rs_draw, scale, shape)
    
    return rs_draw, pr_draw


#==================================================past_stat_finder==============================================

def stat_finder(season_data, game_ind, eps1, eps2, shape, pe_method, t_id):
    """
    finds past stats (at the end of previous game) of a team for a specific "game_ind" in "season_data" 
    Note: t_id = 1 or 2, determines whether the team is in the first or second column of season_data
    """
    team_id  = "team" + str(t_id)
    game_id  = team_id + "_gn"

    team_name = season_data.loc[game_ind, team_id]
    team_gnum = season_data.loc[game_ind, game_id]


    if  team_gnum == 1:
        past_stats = team_stats_initial(team_name, shape)
    else:
        row_id, t_id = game_finder(season_data, team_name, team_gnum - 1)
        past_stats   = team_stats(t_id, row_id, season_data, eps1, eps2, shape, pe_method)
    
    return past_stats


#==================================================team-stats_initial==============================================

def team_stats_initial(team_name, shape):
    """"
    sets the initial values of all relevant stats for a team at the beginning of the season!
    """
      
    stats =  {"team"    : team_name,
              "scale"   : 6.0,
              "shape"   : shape,
              "past_rs" : 0,
              "past_ra" : 0,
              "game_n"  : 1,
              "tol"     : 1.0,  
              "pe"      : 0.50                 
              }
    
    return stats


#==================================================team-stats==============================================

def team_stats(t_id, row_ind, season_data, eps1, eps2, shape, pe_method):
    """"
    extract relevant stats for a "t_id" (team1 or team2) in a "row_ind" given "season_data" and parameters
    game_ind is the index of season_data that contains the information of this team at the end of its previous match!
    """
    team_id  = "team" + str(t_id)
    past_rs  = team_id + "_past_rs"
    past_ra  = team_id + "_past_ra"
    game_id  = team_id + "_gn"

    
    stats =  {"team"    : season_data.loc[row_ind, team_id],
              "scale"   : scale_param(season_data.loc[row_ind, past_rs]/season_data.loc[row_ind, game_id], shape),
              "shape"   : shape,
              "past_rs" : season_data.loc[row_ind, past_rs],
              "past_ra" : season_data.loc[row_ind, past_ra],
              "game_n"  : season_data.loc[row_ind, game_id],
              "tol"     : eps1/(season_data.loc[row_ind, game_id]**eps2),  
              "pe"      : PE(season_data.loc[row_ind, past_rs], season_data.loc[row_ind, past_ra],
                                                              season_data.loc[row_ind, game_id], method = pe_method)                 
              }
    
    return stats


#====================================================game_finder============================================

def game_finder(season_data, team_name, game_numb):
    
    """
    finds the index of "season_data" where "team_name" played its "game_numb"-th match 
    """  

    set1 = season_data[(season_data['team1'] == team_name)    | (season_data['team2'] == team_name)].index
    set2 = season_data[(season_data['team1_gn'] == game_numb) | (season_data['team2_gn'] == game_numb)].index
    
    
    for row_index in set1.intersection(set2):
        if (season_data.loc[row_index, 'team1'] == team_name) & (season_data.loc[row_index, 'team1_gn'] == game_numb):
                r_id = row_index
                t_id = 1
                break

        elif (season_data.loc[row_index, 'team2'] == team_name) & (season_data.loc[row_index, 'team2_gn'] == game_numb):
                r_id = row_index
                t_id  = 2
                break            
            
    
    return r_id, t_id
                

    
#====================================================simulate_season===============================================
   
    
def season_sim(season_data, scale_bayes, n_sim, eps1, eps2, shape, pe_method, sim_method, opp_param):
    """
    simulates the entire season
    """
    sim_g_array = np.zeros(len(season_data.index))
    rej_array = np.zeros(len(season_data.index))
    for game_ind in range(len(season_data.index)): 
        # 1) identify teams in this game and find their stats at the end of their last game
        team_1 = stat_finder(season_data, game_ind, eps1, eps2, shape, pe_method, 1)
        team_2 = stat_finder(season_data, game_ind, eps1, eps2, shape, pe_method, 2)

        # 2) simulation
        sim_g, rej = simulation_weibull(n_sim, team_1, team_2, scale_bayes, shape, opp_param, pe_method, sim_method)
        sim_g_array[game_ind] = sim_g[sim_g["res(team_1)"]==1]["prob"].sum()
        rej_array[game_ind] = rej

    return sim_g_array, rej_array


#====================================================simulation_weibull============================================

def simulation_weibull(n_sim, team_1, team_2, scale_bayes, shape, opp_param, pe_method, sim_method):
    
    """
    simulates "n_sim" number of games given teams' stats
    returns a dataframe of "accepted" predictions
    """
    # {"base", "bayes", "opp", "both"}
    
    shape_factor = math.gamma(1+(1.0/shape))
    
    sim_g = pd.DataFrame(columns = ['rs_1', 'rs_2', 'pr_g', 'pe_1', 'ppe_1', 'pe_2', 'ppe_2', 'prob', 'res(team_1)'],
                                                                                                index = np.arange (0, n_sim))
    sim_g["pe_1"] = team_1["pe"]
    sim_g["pe_2"] = team_2["pe"]
    

    team_1_scale = team_1["scale"]
    team_2_scale = team_2["scale"]
    
    if sim_method == "bayes" :
        
        team_1_scale = scale_bayes.loc[team_1["game_n"]-1, team_1["team"]]
        team_2_scale = scale_bayes.loc[team_2["game_n"]-1, team_2["team"]]
                
    elif sim_method == "opp" :
        
        if (team_2["past_ra"]/team_2["game_n"]) < team_1_scale :
            
            team_1_scale = opp_param * team_1_scale + opp_param * (team_2["past_ra"]/team_2["game_n"])
           # team_1_scale = opp_param * team_1_scale
        
        
        if (team_1["past_ra"]/team_1["game_n"]) < team_2_scale:
            
            team_2_scale = opp_param * team_2_scale + opp_param * (team_1["past_ra"]/team_1["game_n"])
          # team_2_scale = opp_param * team_2_scale
        
    elif sim_method == "both" :
        
        team_1_scale = scale_bayes.loc[team_1["game_n"]-1, team_1["team"]]
        team_2_scale = scale_bayes.loc[team_2["game_n"]-1, team_2["team"]]
        
        if (team_2["past_ra"]/team_2["game_n"]) < team_1["scale"] :
            
            team_1_scale = ((team_1_scale*shape_factor + (team_2["past_ra"]/team_2["game_n"]))/2.0) * (1.0/shape_factor)
           # team_1_scale = opp_param * team_1_scale
        
        
        if (team_1["past_ra"]/team_1["game_n"]) < team_2["scale"]:
            
            team_2_scale = ((team_2_scale*shape_factor + (team_1["past_ra"]/team_1["game_n"]))/2.0) * (1.0/shape_factor)
          # team_2_scale = opp_param * team_2_scale
         

    
    i = 0
    j = 0   
    
    while (i < n_sim):
            
     # runs for each team in this simulation        
        rs_1_now, pr_1_now = runs_scored_weibull(team_1_scale, team_1["shape"])
        rs_2_now, pr_2_now = runs_scored_weibull(team_2_scale, team_2["shape"])
        
    # cumulative runs and predicted PE for each team at the end of the game    
        cum_rs_1 = team_1["past_rs"] + rs_1_now
        cum_rs_2 = team_2["past_rs"] + rs_2_now

        cum_ra_1 = team_1["past_ra"] + rs_2_now
        cum_ra_2 = team_2["past_ra"] + rs_1_now

        ppe_1 = PE(cum_rs_1, cum_ra_1, team_1["game_n"] + 1, method = pe_method)
        ppe_2 = PE(cum_rs_2, cum_ra_2, team_2["game_n"] + 1, method = pe_method)


        # is this game acceptable? yes, if for both PPEs |PPE - PE| < tol  

        if (np.abs(ppe_1 - team_1["pe"]) <= team_1["tol"]) and (np.abs(ppe_2 - team_2["pe"]) <= team_2["tol"]):

            sim_g.loc[i, "rs_1"]  = rs_1_now
            sim_g.loc[i, "rs_2"]  = rs_2_now
            sim_g.loc[i, "pr_g"]  = pr_1_now * pr_2_now
            sim_g.loc[i, "ppe_1"] = ppe_1
            sim_g.loc[i, "ppe_2"] = ppe_2

            if (rs_1_now > rs_2_now):
                sim_g.loc[i, "res(team_1)"] = 1
            elif (rs_1_now == rs_2_now):
                sim_g.loc[i, "res(team_1)"] = 0
            else:
                sim_g.loc[i, "res(team_1)"] = -1

            i += 1
        else:
            j += 1


    sim_g["prob"] = sim_g.loc[:, "pr_g"]/sim_g["pr_g"].sum()   # adjusted probability
    sim_g["prob"] = sim_g["prob"]/sim_g["prob"].sum()

    return sim_g, j

#====================================================Brier_Score============================================

def bscore(predicted, observed):
    """
    Computes Brier score (between 0 and 1), with zero being most accurate
    Note: "observed" is binary with 1 indicating win
    Note: "predicted" may be probabilistic with each number indicating Pr(win) consistent with a 1 in "observed"
    """
    return (sum((predicted - observed) * (predicted - observed)))/len(predicted)


