# Pythagorean Expectation as a Predictor for Baseball Games

### Table of Contents
1. [Project Overview](#overview)
2. [Data Gathering](#data-gathering)
3. [Our Models](#models)
     1. [Pythagorean Model](#pythagorean)
     2. [ELO Model](#elo)
4. [Comparisons of Models](#comparison)

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
(see [Pythagorean Modeal](#pythagorean)). 

As for a comparison of our model, we saw that 538 has an [ELO model](https://github.com/fivethirtyeight/data/tree/master/mlb-elo)
which at each point in the season gives each team an ELO rating. We decided that this would be a good model to compare our Pythagorean
model to. We also thought that there might be some additional factors that we might consider to enhance the 538 model such as the 
pitcher or home field advantage (see [ELO Model](#elo)).

After creating our Pythagorean model and the different ELO based models that we consider from 538, we then compare them using the 
Brier score which is a metric that measures how well the prediction of an event is to the actuality of that said event (see [Comparions of Models](#comparison)).

## Data gathering <a name = "data-gathering"></a>
### Data Sources
For our project, we used the data from [retrosheet](https://www.retrosheet.org/) specifically we 
focused solely on the years from 2010-2022. As mentioned above, we found that 538 had an [ELO model](https://github.com/fivethirtyeight/data/tree/master/mlb-elo) which we also had copied, but we use
this data for comparison purposes as to compare our model to their model. 

### Preprocessing Data

The data from retrosheet was given in a txt file. Furthermore, their data consisted of a multitude of 
variables which we were not interested in for our Pythagorean model such as individual players statistics,
umpire statistics, weather, etc. Thus, we took only the required statistics from retrosheet, and created a
csv file where we also added the additional statistics of their win percentage and Pythagorean expecation. 
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

For completeness, we note that we have 538's ELO data stored in the csv files in the [mlb_elo_data](https://github.com/schulze61/erdos_baseball_project/tree/main/mlb_elo_data) folder. 

## Our Models <a name = "models"></a>

### Pythagorean Model <a name = "pythagorean"></a>

### ELO model <a name = "elo"></a>

## Comparison of Models <a name = "comparison"></a>

### Brier Score
The metric which we decided to use to measure the effectiveness of each of the models will be the Brier score. This score is defined by the following formula
$$\text{Brier Score} = \frac{1}{N}\sum_{i=1}^N(p_i-o_i)^2$$
where $p_i$ is the the probability that we have predicted for a team to win in game $i$ and $o_i$ is the outcome of game $i$ (i.e. $1$ if it is a win and $0$ if it is a loss). The main observation about the Brier scored which we must note is that a low Brier score is better (we see this if a team is $p_i=1$ and $o_i=1$, then we predicted with 100% accuracy that we would win and we were correct, and this contributes $0$ to the sum decreasing the score. On the other hand if $p_i=1$ and $o_i=0$, then we predicted with certainty a win, but a loss occured, so we were wrong, and this would contribute $1$ to the sum increasing the score). 

We compared the Brier scores of our Pythagorean model and the different ELO models, and we plotted them against the years to see how the models compare. We insert the graph below for the reader. 

