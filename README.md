# Pythagorean Expectation as a Predictor for Baseball Games

### Table of Contents
1. [Project Overview](#overview)
2. [Data Gathering](#data-gathering)
3. [Our Models](#models)
     1. [Pythagorean Models](#pythagorean) 

        1. [Basic Model](#basic)
        2. [Bayes Model](#bayes)
        3. [Opponent Model](#opp)
        4. [Both Model](#both)
     2. [Elo Model](#elo)


## Project Overview <a name = "overview"></a>

In 1983, Bill James came up with a simple formula to predict the win percentage of a baseball team by 
only looking at the runs scored and the runs allowed for that team. The formula is known as the 
Pythagorean Expectation is given by 
$$\text{Win Percentage} \approx \frac{\text{Runs Scored}^2}{\text{Runs Scored}^2 + \text{Runs Allowed}^2}.$$

The namesake of the formula is given by the resemblance to the Pythagorean Theorem. However, most sabermetricians
agree that taking an exponent of $1.82$ rather than $2$ leads to a slightly more accurate formula. 

Our goal was to build a predictive model for baseball games using the Pythagorean expectation as our main metric 
of a predictor. Our thought process is that if we are in the middle of the season, and we have the win percentages
of the two teams playing, then we should expect whatever the outcome of the game to be such that our Pythagorean
expectations tend towards the new win ratios of both teams. However, when doing our research into this, we found
a [paper](https://arxiv.org/pdf/math/0509698.pdf) by Stephen J. Miller which gives a theoretical justification of 
the Pythagorean expectation formula. The key insight of this paper is that if it is assumed that the runs scored 
and the runs allowed for a given team follow that of a Weibull distribution, then the expected win percentage will
be precisely the Pythagorean expectation. Thus, we decided that since the Pythagorean expectation is a good approximation
of the win percentage, then we may assume that the runs scored and runs allowed of each team follows that of a Weibull
distribution (whose parameters are determined by the average number of runs scored and runs allowed). 
This is the key idea behind our first predictive model which shall be explained in further detail below 
(see section [Pythagorean Modeal](#pythagorean)). 

As for a comparison of our model, we saw that 538 has an [Elo model](https://github.com/fivethirtyeight/data/tree/master/mlb-elo)
which at each point in the season gives each team an Elo rating. We decided that this would be a good model to compare our Pythagorean
model to. We note that 538 gave two separate probabilities of winning based on both the Elo score and additional factors (see section [Elo Model](#elo)).

### Brier Score

The metric that we decided to use to measure the effectiveness of each of the models
will be the Brier score. This score is defined by the following formula
$$\text{Brier Score} = \frac{1}{N}\sum_{i=1}^N(p_i-o_i)^2$$
where $p_i$ is the probability that we have predicted for a team to win in game $i$ and $o_i$ is the outcome of game $i$ (i.e. $1$ if it is a win and $0$ if it is a loss). The main observation about the Brier score which we must note is that a low Brier score is better (we see this if a team is $p_i=1$ and $o_i=1$, then we predicted with 100% accuracy that we would win and we were correct, and this contributes $0$ to the sum decreasing the score. On the other hand if $p_i=1$ and $o_i=0$, then we predicted with certainty a win, but a loss occured, so we were wrong, and this would contribute $1$ to the sum increasing the score). 


After creating our Pythagorean model and the different Elo based models that we consider from 538, we compare them using the 
Brier score which is a metric that measures how well the prediction of an event is to the actuality of that said event (see section [Comparion of Models](#comparison)).

## Data gathering <a name = "data-gathering"></a>
### Data Sources
For our project, we used the data from [retrosheet](https://www.retrosheet.org/) specifically we 
focused solely on the years from 2010-2022. As mentioned above, we found that 538 had an [Elo model](https://github.com/fivethirtyeight/data/tree/master/mlb-elo) which we also had copied, but we used
this data for comparison purposes to compare our model to their model. 

### Preprocessing Data

The data from retrosheet was given in a txt file. Furthermore, their data consisted of a multitude of 
variables that we were not interested in for our Pythagorean model such as individual players statistics,
umpire statistics, weather, etc. Thus, we took only the required statistics from retrosheet, and created a
csv file where we also added the additional statistics of their win percentage and Pythagorean expectation. 
(see [Sort_game_logs_v2.ipynb](https://github.com/schulze61/erdos_baseball_project/blob/main/game_log_sorter/Sort_game_logs_v2.ipynb))
This initial data cleaning with the csv files we have stored in the [sorted_game_logs](https://github.com/schulze61/erdos_baseball_project/tree/main/game_log_sorter/sorted_game_logs) folder. 


After doing some preliminary work on our model, we did some additional data cleaning since the retrosheet
data as is had each game listed twice (one time for each team playing). Thus, to avoid redundancy, we cut 
the data in half so each game only appeared once, but this was at the cost of creating additional columns
and distinuguishing between a Team_1 and a Team_2. Furthermore, at this step we also added additional 
data into the csv file that would be useful in the perspective of modeling the runs scored and runs allowed
in terms of the Weibull distribution. Thus, we added the cumulative and average runs scored and allowed for 
each team prior to the given game, so we have how they did before the given game (see [Resort_Data_v3.ipynb](https://github.com/schulze61/erdos_baseball_project/blob/main/game_log_sorter/Resort_Data_v3.ipynb)). The
csv files for the second cleaning of the data are stored in the [new_sorted_game_logs](https://github.com/schulze61/erdos_baseball_project/tree/main/game_log_sorter/new_sorted_game_logs) folder. 

For completeness, we note that we have 538's Elo data stored in the csv files in the [mlb_elo_data](https://github.com/schulze61/erdos_baseball_project/tree/main/mlb_elo_data) folder. 

## Our Models <a name = "models"></a>

### Pythagorean Models <a name = "pythagorean"></a>
We remark that this section is the main focus as it is the model that we created. As described in the introduction, our model is based off of the Pythagorean expectation. We shall denote the Pythagorean expectation of a team at game $n$ by 
$PE_n$. For a given team, we make the assumption that the change of the Pythagorean expectation is not too large, that is there will be two constants $\epsilon_1$ and $\epsilon_2$ such that 
$$\vert PE_{n+1}-PE_n\vert\leq \frac{\epsilon_1}{n^{\epsilon_2}}.$$
Equivalently, we will have that 
$$\ln\left(\left\vert PE_{n+1}-PE_n\right\vert\right)\leq -\epsilon\ln(n)$$
The purpose of these quantities of $\epsilon_1$ and $\epsilon_2$ is that when we run our simulations of games, we don't expect 
for the Pythagorean expectation to change too much after a given game, and it is more variable to change at the beginning of 
the season (i.e. when $n$ is small) than later in the season. Thus, when we simulate a game between two teams we will discard 
any simulations where the change in Pythagorean expectation exceeds this tolerated error.

One of our main goals was to discover which values of $\epsilon_1$ and $\epsilon_2$ will optimize our model. We had two methods 
to solve for these values we did a regression, and we made a heat map. We outline both methods below; however, we remark that in the final model, we used the heat map and ran our model for the extreme values on the heat map to finalize these values. 

#### Regression

For our regression, we chose a single team and plotted the change of the Pythagorean expectation of that team over the games of a 
single season. After doing this for a single team, we were curious if each team would have roughly the same shape, so we overlayed graphs for each teams change of Pythagorean expectation over the number of games in the season and got the following graph:

<p align="center">
  <img src="https://github.com/schulze61/erdos_baseball_project/blob/main/Change%20in%20PE%20for%20all%20teams%20by%20Game%20Number.png" />
</p>

We noticed that this graph tends to be well approximated by $\frac{\epsilon_1}{n^{\epsilon_2}}$, and each team had roughly the same
shape, so we decided to just choose a fixed $\epsilon_1\approx 1$ and $\epsilon_2\approx 1.4237$ for all simulations. Our first goal to determine these 
values was to run a linear regression. This gave us some preliminary values, but we wanted to optimize these values with respect 
to the Brier score (see section [Comparison of Models](#comparison)). This led us to create a heat map for different values of $\epsilon_1$ and $\epsilon_2$ showing the different Brier scores.

#### Heat Map
We thus created the following heat map where the axes correspond to the choices of $\epsilon_1$ and $\epsilon_2$, and the heat 
measures the Brier score for running a smaller simulation with these values of $\epsilon_1$ and $\epsilon_2$. We noticed that there 
appeared to be a line where below this line the Brier score tended to be below $.25$ which is the goal. Furthermore, the black dot corresponds to our initial choice of $\epsilon_1$ and $\epsilon_2$ from the regression. We had this was a Brier score of around $.27$, but we saw that reducing these parameters should decrease our Brier score.

<p align="center">
  <img src="https://github.com/schulze61/erdos_baseball_project/blob/main/Heat%20Map%20for%20Epsilon.png" style="background-color:white;"/>
</p>

As there appeared to be this boundary where the Brier score was lower, we decided to run a full simulation for the extreme values of  $\epsilon_1$ and $\epsilon_2$ and different simulation methods to determine what will be the optimal choices. We also noted that there appeared to be small patches that appeared to be better in this region; however, we chose to disregard this as we created this heat map using smaller simulations. From many simulations, we found that there was not a significant change in the Brier score for different choices of $\epsilon_1$ and $\epsilon_2$ under the line. Thus, we settled upon the choice that $\epsilon_1=1.5$ and $\epsilon_2 = .5$. 

We will now go into our different methods of simulation and which we took to be the most optimal. 

### Methods of Simulation

We now explain the four different ways that we used the Pythagorean expectation to simulate games. We call them Basic, Bayes, Opponent, and Both. The Basic model will be the more naive thing, the Bayes model will add by incorporating some Bayesian techniques, the opponent model compares the opponent's defense to the offense, and lastly, the Both model combines the Bayes and Opponent models. We detail these models below. Note: to see the code that defines these simulations see [pemod.py](https://github.com/schulze61/erdos_baseball_project/blob/main/pemod.py).

#### Basic Method <a name = "basic"></a>

For the basic model, we had made the assumption that the runs scored and runs allowed follow a Weibull distribution whose parameter can be determined by the average number of runs scored and allowed for each team. After calculating the parameters for these distributions for each team for each game, we then simulate these distributions to estimate the runs scored. In doing this, we count how many of our simulations each team wins and use that to give a probability that said team will win that game. This is the method of simulation for these four models; however, the different models will have different methods to determine the parameters of the Weibull distributions. 


#### Bayes Model <a name = "bayes"></a>

We assume that runs scored by each team are both random variables that follow a Weibull distribution. We follow the scientific research in the field to determine the shape parameter of the underlying Weibull distribution. Having fixed that parameter, in the first simulation, we assume that the scale paramater of the underlying Weibull distribution for each team is equal to the average runs that they have scored through their last game. In the second simulation, we use an alternative approach where this scale parameter for each team is determined through a continuous `Bayesian` updating process. The general idea is that we form a `prior` belief about the expected runs that a team can score at the beginning of the season. Then, as the season progresses, by observing the number of runs that they score in each game, we `update` our belief about their future peroformance. This updated belief, is then, used for simulation and prediction of their next game.

We rely on statistical theory to form this Bayeisan updating algorithm. The theory tells us that if a random variable $x$ has a Weibull distribution with a *known* shape, then, an `Inverse Gamma (IG)` distribution would be a `conjugate` prior distribution for its scale parameter. In other words, consider the following Weibull random variable

$$ f(x|k, \theta) = \frac{k}{\theta} x^{k-1} e^{-\frac{x^k}{\theta}}$$

where $k$ is the **known** shape and $\\theta$ is the **unknown** scale parameter. If we assume the the scale parameter $\\theta$, is a random variable with an IG distribution with parameters *a* and *b*, we can formulate the problem in such a way that after each observation only the parameters of the IG distribution are updated. In this framework, let's assume that we start with a prior distribution with parameters $a_0$ and $b_0$ ($k$ is the known shape parameter), i.e. 

$$\\theta|k \\sim IG(a_0, b_0) $$

Then, if we observe $n$ games where the team in each game scores $rs_i \\hspace{2mm}$ runs for $i \\in \\{1,2, \\ldots, n\\}$, then, we can form our updated `posterior` belief as

$$\\theta|k \\sim IG(a_n, b_n) $$

where

$$a_n = a_0 + n $$

and 

$$ b_n = b_0 + \\sum\\limits_{i=1}^{n} rs_i^{k}$$

This is the algorithm we have followed in our simulation. The only issue is in the implementation of the algorithm as in Python the Weibull distribution is defined with a slightly different notation. Specifically, in Python a Weibull random variable has the following PDF:

$$f(x|k,\\lambda) = \\frac{k}{\\lambda} (\\frac{x}{\\lambda})^{(k-1)} e^{-(\\frac{x}{\\lambda})^k}$$

We should note that the two notations are identical iff $\\theta = \\lambda^k$. Using this relation, we formulate our Bayesian updating algorithm. 



The above discussion completes the algorithm provided that we have a known **prior** i.e., $a_0$ and $b_0$*. In order to form our prior we make two assumptions. The first, is a standard assumption, $ a_0 = 2$. Note that, in the first notation,

$$\\theta|k \\sim IG(a_0, b_0) \\hspace{3mm} \\Rightarrow \\hspace{3mm} \\mathbb{E}[\theta] = \\frac{b_0}{a_0-1} $$

This normalizing assumption basically implies that $\mathbb{E}[\theta] = b_0$.

In order to determine $b_0$, then, we can rely on the observed data. The initial values of the hyperparameter are defined such that before the first game, all teams are expected to score `rs_0` runs in their first game. This is an *uninformed prior*. Since the runs scored have a Weibull distribution:

$$\\mathbb{E}[\text{runs}] = \\lambda \\Gamma(1+1/k)  \\hspace{3mm} \\Rightarrow \\hspace{3mm} \\lambda_0 = \\frac{rs_0}{\\Gamma(1+1/k)}$$

Thus,

$$\\mathbb{E}[\lambda_0^k] = \\frac{b}{a-1}  \\hspace{3mm} \\Rightarrow \\hspace{3mm}\\Big[\\frac{rs_0}{\\Gamma(1+1/k)} \\Big]^k = \\frac{b}{2-1}$$

This gives us the initial hyperparameters for all teams :

$$ b_0 = \\Big[\\frac{rs_0}{\\Gamma(1+1/k)} \\Big]^k  \\hspace{10mm} \text{and } \\hspace{10mm} a_0 = 2.0$$

After one game, if a team scores $rs_1$ rusn. Then, we update our belief about their expected future runs as the following: 

$$ \\lambda^k \\sim IG(a_0 +1, b_0 + rs_1^{k})$$

Considering that,

$$\\mathbb{E}[\lambda^k] = \\frac{ b_0 + rs_1^{k}}{a_0 +1 -1} $$

we have:

$$\\lambda_1 = \\Big[\\frac{ b_0 + rs_1^{k}}{a_0 +1 -1} \\Big]^{\\frac{1}{k}}$$

This is the scale parameter that we use in our simulation. Similarly, after n games:

$$\\lambda_n = \left[\frac{ b_0 + \sum\limits_{i=1}^{n} rs_i^{k}}{a_0 + n -1}\right]^{\\frac{1}{k}}$$

This is how we do our Bayesian updating for the scale parameter for the Bayes model. 

#### Opponent Model <a name = "opp"></a>

For the Opponent Model, we take into account the defense of the opposing team in the following manner. We see if the average number of runs scored is larger than the average number of runs allowed for the opposing team, we might expect the offense to do better, so we increase the scale parameter in this case. 


#### Both Model  <a name = "both"></a>

For the Both Model, we first use our Bayesian updating method to get the parameters, and then we will do our comparison of the offense against the defense of each team, and make the adjustments as in the Opponent Model before we run the simulation. 

#### Comparison of These Models

Below we have a graph comparing the Brier scores of each of the four above models over the years 2010-2022. We remark that the spike in the year 2020 is likely due to the season being cut short due to Covid since we expect our model to be more accurate the for the further along in the season that we are. 

### Elo model <a name = "elo"></a>

We note that for the Elo model, we just gathered the data from 538, and for each game 
538 gave two separate probabilities for the team to win. The first probability takes 
only into account the Elo scores of both teams to give a probability of a team 
winning. We call this the "Elo model". 

The second model that 538 gives is a probability that takes into account both the Elo
scores of both teams, but it also takes into account the starting pitchers of both 
teams to give an adjusted score. This model we call that "538 Rating". 
