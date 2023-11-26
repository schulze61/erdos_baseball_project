# Pythagorean Expectation as a Predictor for Baseball Games

### Table of Contents
1. [Project Overview](#overview)
2. [Data Gathering](#data-gathering)
3. [Our Two Models](#models)
     1. [Pythagorean Model](#pythagorean)
     2. [ELO Model](#elo)
4. [Comparisons with Existing Models](#comparison)

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

We also had another idea 






## Data gathering <a name = "data-gathering"></a>
### Data Sources
For our project, we used the data from [retrosheets](https://www.retrosheet.org/) specifically we 
focused solely on the years from 2010-2022. As retrosheets had a 



