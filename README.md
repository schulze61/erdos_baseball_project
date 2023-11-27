# Pythagorean Expectation as a Predictor for Baseball Games

### Table of Contents
1. [Project Overview](#overview)
2. [Data Gathering](#data-gathering)
3. [Our Models](#models)
     1. [Pythagorean Model](#pythagorean)
     2. [Elo Model](#elo)
4. [Comparison of Models](#comparison)

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
This is the key idea behind our first predicitive model which shall be explained in further detail below 
(see section [Pythagorean Modeal](#pythagorean)). 

As for a comparison of our model, we saw that 538 has an [Elo model](https://github.com/fivethirtyeight/data/tree/master/mlb-elo)
which at each point in the season gives each team an Elo rating. We decided that this would be a good model to compare our Pythagorean
model to. We note that 538 gave two separate probabilities of winning based on both the Elo score and additional factors (see section [Elo Model](#elo)).

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

### Pythagorean Model <a name = "pythagorean"></a>
We remark that this section is the main focus as it is the model that we created. As described in the introduction, our model is based off of the Pythagorean expectation. We shall denote the Pythagorean expectation of a team at game $n$ by 
$PE_n$. For a given team, we make the assumption that the change of the Pythagorean expectation is not too large, that is there will be two constants $\epsilon_1$ and $\epsilon_2$ such that 
$$\vert PE_{n+1}-PE_n\vert\leq \frac{\epsilon_1}{n^{\epsilon_2}}.$$
Equivalently, we will have that 
$$\ln\left(\left\vert PE_{n+1}-PE_n\right\vert\right)\leq \ln(\epsilon_1)-\epsilon_2\ln(n)$$

### Elo model <a name = "elo"></a>

We note that for the Elo model, we just gathered the data from 538, and for each game 
538 gave two separate probabilities for the team to win. The first probability takes 
only into account the Elo scores of both teams to give a probability of a team 
winning. We call this the "Elo model". 

The second model that 538 gives is a probability that takes into account both the Elo
scores of both teams, but it also takes into account the starting pitchers of both 
teams to give an adjusted score. This model we call that "538 Rating". 

When observing these models, we had made the observation that most of the 
probabilities were around 50%, so a coin flip might as well be how we predict a winner,
but we decided that the home team might have a slight edge on average. Thus, we decided
to make a naive model as well where we always predict that the home team will win with
a probability of $p=.5568$. We call this the "Home Advantage Model". Although we did
not expect this to be a good model, we discovered that it was a bit useful to see in 
which years the home-field advantage had more of an impact. 

## Comparison of Models <a name = "comparison"></a>

### Brier Score
The metric that we decided to use to measure the effectiveness of each of the models
will be the Brier score. This score is defined by the following formula
$$\text{Brier Score} = \frac{1}{N}\sum_{i=1}^N(p_i-o_i)^2$$
where $p_i$ is the probability that we have predicted for a team to win in game $i$ and $o_i$ is the outcome of game $i$ (i.e. $1$ if it is a win and $0$ if it is a loss). The main observation about the Brier score which we must note is that a low Brier score is better (we see this if a team is $p_i=1$ and $o_i=1$, then we predicted with 100% accuracy that we would win and we were correct, and this contributes $0$ to the sum decreasing the score. On the other hand if $p_i=1$ and $o_i=0$, then we predicted with certainty a win, but a loss occured, so we were wrong, and this would contribute $1$ to the sum increasing the score). 

We compared the Brier scores of our Pythagorean model and the different Elo models, and we plotted them against the years to see how the models compare. We insert the graph below for the reader. 

