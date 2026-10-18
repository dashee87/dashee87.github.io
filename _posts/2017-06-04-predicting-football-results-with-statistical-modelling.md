---
title: "Predicting Football Results With Statistical Modelling"
excerpt: "Combining the world's most popular sport with everyone's favourite discrete probability distribution, this post predicts football matches using the Poisson distribution."
layout: single
header:
  overlay_image: football-overlay.jpg
  overlay_filter: 0.4
  caption: ""
  cta_label: "Switch to R version"
  cta_url: "https://dashee87.github.io/data%20science/football/r/predicting-football-results-with-statistical-modelling/"
categories:
  - football
  - python
tags:
  - football
  - soccer
  - epl
  - Poisson
  - Betfair
  - python
author: "David Sheehan"
date: "4 June 2017"
hidden: true
---

Football (or soccer to my American readers) is full of clichés: "It's a game of two halves", "taking it one game at a time" and "Liverpool have failed to win the Premier League". You're less likely to hear "Treating the number of goals scored by each team as independent Poisson processes, statistical modelling suggests that the home team have a 60% chance of winning today". But this is actually a bit of cliché too (it has been discussed [here](https://www.pinnacle.com/en/betting-articles/soccer/how-to-calculate-poisson-distribution), [here](https://help.smarkets.com/hc/en-gb/articles/115001457989-How-to-calculate-Poisson-distribution-for-football-betting), [here](http://pena.lt/y/2014/11/02/predicting-football-using-r/), [here](http://opisthokonta.net/?p=296) and [particularly well here](https://dashee87.github.io/data%20science/football/r/predicting-football-results-with-statistical-modelling/)). As we'll discover, a simple Poisson model is, well, overly simplistic. But it's a good starting point and a nice intuitive way to learn about statistical modelling. So, if you came here looking to make money, [I hear this guy makes £5000 per month without leaving the house](http://www.make5000poundspermonth.co.uk/).

## Poisson Distribution

The model is founded on the number of goals scored/conceded by each team. Teams that have been higher scorers in the past have a greater likelihood of scoring goals in the future. We'll import all match results from the recently concluded Premier League (2016/17) season. There's various sources for this data out there ([kaggle](https://www.kaggle.com/hugomathien/soccer), [football-data.co.uk](http://www.football-data.co.uk/englandm.php), [github](https://github.com/jalapic/engsoccerdata), [football-data.org API](https://www.football-data.org/) (or the keyless [Football Charts JSON API](https://www.football-charts.com/developers))). I built an [R wrapper for that API](https://github.com/dashee87/footballR), but I'll go the csv route this time around. 


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam

epl_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E0.csv")
epl_1617 = epl_1617[['HomeTeam','AwayTeam','FTHG','FTAG']]
epl_1617 = epl_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1617.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>HomeGoals</th>
      <th>AwayGoals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Burnley</td>
      <td>Swansea</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Crystal Palace</td>
      <td>West Brom</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Everton</td>
      <td>Tottenham</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hull</td>
      <td>Leicester</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Man City</td>
      <td>Sunderland</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We imported a csv as a pandas dataframe, which contains various information for each of the 380 EPL games in the 2016-17 English Premier League season. We restricted the dataframe to the columns in which we're interested (specifically, team names and numer of goals scored by each team). I'll omit most of the code that produces the graphs in this post. But don't worry, you can find that code on [my github page](https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-06-04-predicting-football-results-with-statistical-modelling.ipynb). Our task is to model the final round of fixtures in the season, so we must remove the last 10 rows (each gameweek consists of 10 matches).


```python
epl_1617 = epl_1617[:-10]
epl_1617.mean()
```




    HomeGoals    1.591892
    AwayGoals    1.183784
    dtype: float64



You'll notice that, on average, the home team scores more goals than the away team. This is the so called 'home (field) advantage' (discussed [here](https://jogall.github.io/2017-05-12-home-away-pref/)) and [isn't specific to soccer](http://bleacherreport.com/articles/1803416-is-home-field-advantage-as-important-in-baseball-as-other-major-sports). This is a convenient time to introduce the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution). It's a discrete probability distribution that describes the probability of the number of events within a specific time period (e.g 90 mins) with a known average rate of occurrence. A key assumption is that the number of events is independent of time. In our context, this means that goals don't become more/less likely by the number of goals already scored in the match. Instead, the number of goals is expressed purely as function an average rate of goals. If that was unclear, maybe this mathematical formulation will make clearer:

$$
P\left( x \right) = \frac{e^{-\lambda} \lambda ^x }{x!}, \lambda>0
$$

$$\lambda$$ represents the average rate (e.g. average number of goals, average number of letters you receive, etc.). So, we can treat the number of goals scored by the home and away team as two independent Poisson distributions. The plot below shows the proportion of goals scored compared to the number of goals estimated by the corresponding Poisson distributions.


<div style="text-align:center" markdown="1">

![]({{ base_path }}/images/home_away_goals_python.png)

</div>


We can use this statistical model to estimate the probability of specfic events.

$$
\begin{align*}
P(\geq 2|Home) &= P(2|Home) + P(3|Home) + ...\\
        &= 0.258 + 0.137 + ...\\
        &= 0.47
\end{align*}
$$

The probability of a draw is simply the sum of the events where the two teams score the same amount of goals.

$$
\begin{align*}
P(Draw) &= P(0|Home) \times P(0|Away) + P(1|Home) \times P(1|Away) + ...\\
        &= 0.203 \times 0.306 + 0.324 \times 0.362 + ...\\
        &= 0.248
\end{align*}
$$

Note that we consider the number of goals scored by each team to be independent events (i.e. P(A n B) = P(A) P(B)). The difference of two Poisson distribution is actually called a [Skellam distribution](https://en.wikipedia.org/wiki/Skellam_distribution). So we can calculate the probability of a draw by inputting the mean goal values into this distribution.


```python
# probability of draw between home and away team
skellam.pmf(0.0,  epl_1617.mean()[0],  epl_1617.mean()[1])
```




    0.24809376810717076




```python
# probability of home team winning by one goal
skellam.pmf(1,  epl_1617.mean()[0],  epl_1617.mean()[1])
```




    0.22558259663675409



<div style="text-align:center" markdown="1">

![]({{ base_path }}/images/skellam_goals_python.png)

</div>

So, hopefully you can see how we can adapt this approach to model specific matches. We just need to know the average number of goals scored by each team and feed this data into a Poisson model. Let's have a look at the distribution of goals scored by Chelsea and Sunderland (teams who finished 1st and last, respectively).

<div style="text-align:center" markdown="1">

![]({{ base_path }}/images/chelsea_sunderland_goals_python.png)

</div>


## Building A Model

You should now be convinced that the number of goals scored by each team can be approximated by a Poisson distribution. Due to a relatively sample size (each team plays at most 19 home/away games), the accuracy of this approximation can vary significantly (especially earlier in the season when teams have played fewer games). Similar to before, we could now calculate the probability of various events in this Chelsea Sunderland match. But rather than treat each match separately, we'll build a more general Poisson regression model ([what is that?](https://en.wikipedia.org/wiki/Poisson_regression)).

```python
# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

goal_model_data = pd.concat([epl_1617[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl_1617[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>goals</td>      <th>  No. Observations:  </th>  <td>   740</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>GLM</td>       <th>  Df Residuals:      </th>  <td>   700</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Poisson</td>     <th>  Df Model:          </th>  <td>    39</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>log</td>       <th>  Scale:             </th>    <td>1.0</td>  
</tr>
<tr>
  <th>Method:</th>               <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -1042.4</td>
</tr>
<tr>
  <th>Date:</th>           <td>Sat, 10 Jun 2017</td> <th>  Deviance:          </th> <td>  776.11</td>
</tr>
<tr>
  <th>Time:</th>               <td>11:17:38</td>     <th>  Pearson chi2:      </th>  <td>  659.</td> 
</tr>
<tr>
  <th>No. Iterations:</th>         <td>8</td>        <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                 <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th> 
</tr>
<tr>
  <th>Intercept</th>                  <td>    0.3725</td> <td>    0.198</td> <td>    1.880</td> <td> 0.060</td> <td>   -0.016     0.761</td>
</tr>
<tr>
  <th>team[T.Bournemouth]</th>        <td>   -0.2891</td> <td>    0.179</td> <td>   -1.612</td> <td> 0.107</td> <td>   -0.641     0.062</td>
</tr>
<tr>
  <th>team[T.Burnley]</th>            <td>   -0.6458</td> <td>    0.200</td> <td>   -3.230</td> <td> 0.001</td> <td>   -1.038    -0.254</td>
</tr>
<tr>
  <th>team[T.Chelsea]</th>            <td>    0.0789</td> <td>    0.162</td> <td>    0.488</td> <td> 0.626</td> <td>   -0.238     0.396</td>
</tr>
<tr>
  <th>team[T.Crystal Palace]</th>     <td>   -0.3865</td> <td>    0.183</td> <td>   -2.107</td> <td> 0.035</td> <td>   -0.746    -0.027</td>
</tr>
<tr>
  <th>team[T.Everton]</th>            <td>   -0.2008</td> <td>    0.173</td> <td>   -1.161</td> <td> 0.246</td> <td>   -0.540     0.138</td>
</tr>
<tr>
  <th>team[T.Hull]</th>               <td>   -0.7006</td> <td>    0.204</td> <td>   -3.441</td> <td> 0.001</td> <td>   -1.100    -0.302</td>
</tr>
<tr>
  <th>team[T.Leicester]</th>          <td>   -0.4204</td> <td>    0.187</td> <td>   -2.249</td> <td> 0.025</td> <td>   -0.787    -0.054</td>
</tr>
<tr>
  <th>team[T.Liverpool]</th>          <td>    0.0162</td> <td>    0.164</td> <td>    0.099</td> <td> 0.921</td> <td>   -0.306     0.338</td>
</tr>
<tr>
  <th>team[T.Man City]</th>           <td>    0.0117</td> <td>    0.164</td> <td>    0.072</td> <td> 0.943</td> <td>   -0.310     0.334</td>
</tr>
<tr>
  <th>team[T.Man United]</th>         <td>   -0.3572</td> <td>    0.181</td> <td>   -1.971</td> <td> 0.049</td> <td>   -0.713    -0.002</td>
</tr>
<tr>
  <th>team[T.Middlesbrough]</th>      <td>   -1.0087</td> <td>    0.225</td> <td>   -4.481</td> <td> 0.000</td> <td>   -1.450    -0.568</td>
</tr>
<tr>
  <th>team[T.Southampton]</th>        <td>   -0.5804</td> <td>    0.195</td> <td>   -2.976</td> <td> 0.003</td> <td>   -0.963    -0.198</td>
</tr>
<tr>
  <th>team[T.Stoke]</th>              <td>   -0.6082</td> <td>    0.197</td> <td>   -3.094</td> <td> 0.002</td> <td>   -0.994    -0.223</td>
</tr>
<tr>
  <th>team[T.Sunderland]</th>         <td>   -0.9619</td> <td>    0.222</td> <td>   -4.329</td> <td> 0.000</td> <td>   -1.397    -0.526</td>
</tr>
<tr>
  <th>team[T.Swansea]</th>            <td>   -0.5136</td> <td>    0.192</td> <td>   -2.673</td> <td> 0.008</td> <td>   -0.890    -0.137</td>
</tr>
<tr>
  <th>team[T.Tottenham]</th>          <td>    0.0532</td> <td>    0.162</td> <td>    0.328</td> <td> 0.743</td> <td>   -0.265     0.371</td>
</tr>
<tr>
  <th>team[T.Watford]</th>            <td>   -0.5969</td> <td>    0.197</td> <td>   -3.035</td> <td> 0.002</td> <td>   -0.982    -0.211</td>
</tr>
<tr>
  <th>team[T.West Brom]</th>          <td>   -0.5567</td> <td>    0.194</td> <td>   -2.876</td> <td> 0.004</td> <td>   -0.936    -0.177</td>
</tr>
<tr>
  <th>team[T.West Ham]</th>           <td>   -0.4802</td> <td>    0.189</td> <td>   -2.535</td> <td> 0.011</td> <td>   -0.851    -0.109</td>
</tr>
<tr>
  <th>opponent[T.Bournemouth]</th>    <td>    0.4109</td> <td>    0.196</td> <td>    2.092</td> <td> 0.036</td> <td>    0.026     0.796</td>
</tr>
<tr>
  <th>opponent[T.Burnley]</th>        <td>    0.1657</td> <td>    0.206</td> <td>    0.806</td> <td> 0.420</td> <td>   -0.237     0.569</td>
</tr>
<tr>
  <th>opponent[T.Chelsea]</th>        <td>   -0.3036</td> <td>    0.234</td> <td>   -1.298</td> <td> 0.194</td> <td>   -0.762     0.155</td>
</tr>
<tr>
  <th>opponent[T.Crystal Palace]</th> <td>    0.3287</td> <td>    0.200</td> <td>    1.647</td> <td> 0.100</td> <td>   -0.062     0.720</td>
</tr>
<tr>
  <th>opponent[T.Everton]</th>        <td>   -0.0442</td> <td>    0.218</td> <td>   -0.202</td> <td> 0.840</td> <td>   -0.472     0.384</td>
</tr>
<tr>
  <th>opponent[T.Hull]</th>           <td>    0.4979</td> <td>    0.193</td> <td>    2.585</td> <td> 0.010</td> <td>    0.120     0.875</td>
</tr>
<tr>
  <th>opponent[T.Leicester]</th>      <td>    0.3369</td> <td>    0.199</td> <td>    1.694</td> <td> 0.090</td> <td>   -0.053     0.727</td>
</tr>
<tr>
  <th>opponent[T.Liverpool]</th>      <td>   -0.0374</td> <td>    0.217</td> <td>   -0.172</td> <td> 0.863</td> <td>   -0.463     0.389</td>
</tr>
<tr>
  <th>opponent[T.Man City]</th>       <td>   -0.0993</td> <td>    0.222</td> <td>   -0.448</td> <td> 0.654</td> <td>   -0.534     0.335</td>
</tr>
<tr>
  <th>opponent[T.Man United]</th>     <td>   -0.4220</td> <td>    0.241</td> <td>   -1.754</td> <td> 0.079</td> <td>   -0.894     0.050</td>
</tr>
<tr>
  <th>opponent[T.Middlesbrough]</th>  <td>    0.1196</td> <td>    0.208</td> <td>    0.574</td> <td> 0.566</td> <td>   -0.289     0.528</td>
</tr>
<tr>
  <th>opponent[T.Southampton]</th>    <td>    0.0458</td> <td>    0.211</td> <td>    0.217</td> <td> 0.828</td> <td>   -0.369     0.460</td>
</tr>
<tr>
  <th>opponent[T.Stoke]</th>          <td>    0.2266</td> <td>    0.203</td> <td>    1.115</td> <td> 0.265</td> <td>   -0.172     0.625</td>
</tr>
<tr>
  <th>opponent[T.Sunderland]</th>     <td>    0.3707</td> <td>    0.198</td> <td>    1.876</td> <td> 0.061</td> <td>   -0.017     0.758</td>
</tr>
<tr>
  <th>opponent[T.Swansea]</th>        <td>    0.4336</td> <td>    0.195</td> <td>    2.227</td> <td> 0.026</td> <td>    0.052     0.815</td>
</tr>
<tr>
  <th>opponent[T.Tottenham]</th>      <td>   -0.5431</td> <td>    0.252</td> <td>   -2.156</td> <td> 0.031</td> <td>   -1.037    -0.049</td>
</tr>
<tr>
  <th>opponent[T.Watford]</th>        <td>    0.3533</td> <td>    0.198</td> <td>    1.782</td> <td> 0.075</td> <td>   -0.035     0.742</td>
</tr>
<tr>
  <th>opponent[T.West Brom]</th>      <td>    0.0970</td> <td>    0.209</td> <td>    0.463</td> <td> 0.643</td> <td>   -0.313     0.507</td>
</tr>
<tr>
  <th>opponent[T.West Ham]</th>       <td>    0.3485</td> <td>    0.198</td> <td>    1.758</td> <td> 0.079</td> <td>   -0.040     0.737</td>
</tr>
<tr>
  <th>home</th>                       <td>    0.2969</td> <td>    0.063</td> <td>    4.702</td> <td> 0.000</td> <td>    0.173     0.421</td>
</tr>
</table>



If you're curious about the `smf.glm(...)` part, you can find more information [here](http://www.statsmodels.org/stable/examples/notebooks/generated/glm_formula.html) (edit: earlier versions of this post had erroneously employed a Generalised Estimating Equation (GEE)- [what's the difference?](https://stats.stackexchange.com/questions/16390/when-to-use-generalized-estimating-equations-vs-mixed-effects-models)). I'm more interested in the values presented in the `coef` column in the model summary table, which are analogous to the slopes in linear regression. Similar to [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), we take the [exponent of the parameter values](http://www.lisa.stat.vt.edu/sites/default/files/Poisson.and_.Logistic.Regression.pdf). A positive value implies more goals ($$e^{x}>1 \forall x > 0$$), while values closer to zero represent more neutral effects ($$e^{0}=1$$). Towards the bottom of the table you might notice that `home` has a `coef` of 0.2969. This captures the fact that home teams generally score more goals than the away team (specifically, $$e^{0.2969}$$=1.35 times more likely). But not all teams are created equal. Chelsea has a `coef` of 0.0789, while the corresponding value for Sunderland is -0.9619 (sort of saying Chelsea (Sunderland) are better (much worse!) scorers than average). Finally, the `opponent*` values penalize/reward teams based on the quality of the opposition. This relfects the defensive strength of each team (Chelsea: -0.3036; Sunderland: 0.3707). In other words, you're less likely to score against Chelsea. Hopefully, that all makes both statistical and intuitive sense.

Let's start making some predictions for the upcoming matches. We simply pass our teams into `poisson_model` and it'll return the expected average number of goals for that team (we need to run it twice- we calculate the expected average number of goals for each team separately). So let's see how many goals we expect Chelsea and Sunderland to score.


```python
poisson_model.predict(pd.DataFrame(data={'team': 'Chelsea', 'opponent': 'Sunderland',
                                       'home':1},index=[1]))
```




    array([ 3.06166192])




```python
poisson_model.predict(pd.DataFrame(data={'team': 'Sunderland', 'opponent': 'Chelsea',
                                       'home':0},index=[1]))
```




    array([ 0.40937279])



Just like before, we have two Poisson distributions. From this, we can calculate the probability of various events. I'll wrap this in a `simulate_match` function.


```python
def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
simulate_match(poisson_model, 'Chelsea', 'Sunderland', max_goals=3)
```




    array([[ 0.03108485,  0.01272529,  0.00260469,  0.00035543],
           [ 0.0951713 ,  0.03896054,  0.00797469,  0.00108821],
           [ 0.14569118,  0.059642  ,  0.01220791,  0.00166586],
           [ 0.14868571,  0.06086788,  0.01245883,  0.0017001 ]])



This matrix simply shows the probability of Chelsea (rows of the matrix) and Sunderland (matrix columns) scoring a specific number of goals. For example, along the diagonal, both teams score the same the number of goals (e.g. P(0-0)=0.031). So, you can calculate the odds of draw by summing all the diagonal entries. Everything below the diagonal represents a Chelsea victory (e.g P(3-0)=0.149). If you prefer Over/Under markets, you can estimate P(Under 2.5 goals) by summing the entries where the sum of the column number and row number (both starting at zero) is less than 3 (i.e. the 6 values that form the upper left triangle). Luckily, we can use basic matrix manipulation functions to perform these calculations.


```python
chel_sun = simulate_match(poisson_model, "Chelsea", "Sunderland", max_goals=10)
# chelsea win
np.sum(np.tril(chel_sun, -1))
```




    0.8885986612364134




```python
# draw
np.sum(np.diag(chel_sun))
```




    0.084093492686495977




```python
# sunderland win
np.sum(np.triu(chel_sun, 1))
```




    0.026961819942853051



Hmm, our model gives Sunderland a 2.7% chance of winning. But is that right? To assess the accuracy of the predictions, we'll compare the probabilities returned by our model against the odds offered by the [Betfair exchange](https://www.betfair.com/exchange/plus/football).

## Sports Betting/Trading

Unlike traditional bookmakers, on betting exchanges (and Betfair isn't the only one- it's just the biggest), you bet against other people (with Betfair taking a commission on winnings). It acts as a sort of stock market for sports events. And, like a stock market, due to the [efficient market hypothesis](https://en.wikipedia.org/wiki/Efficient-market_hypothesis), the prices available at Betfair reflect the true price/odds of those events happening (in theory anyway). Below, I've posted a screenshot of the Betfair exchange on Sunday 21st May (a few hours before those matches started).

<div style="text-align:center" markdown="1">

![]({{ base_path }}/images/betfair_exchange.png)

</div>

The numbers inside the boxes represent the best available prices and the amount available at those prices. The blue boxes signify back bets (i.e. betting that an event will happen- going long using stock market terminology), while the pink boxes represent lay bets (i.e. betting that something won't happen- i.e. shorting). For example, if we were to bet £100 on Chelsea to win, we would receive the original amount plus 100\*1.13= £13 should they win (of course, we would lose our £100 if they didn't win). Now, how can we compare these prices to the probabilities returned by our model? Well, decimal odds can be converted to the probabilities quite easily: it's simply the inverse of the decimal odds. For example, the implied probability of Chelsea winning is 1/1.13 (=0.885- our model put the probability at 0.889). I'm focusing on decimal odds, but you might also be familiar with [Moneyline (American) Odds](https://www.pinnacle.com/en/betting-articles/educational/odds-formats-available-at-pinnacle-sports) (e.g. +200) and fractional odds (e.g. 2/1). The relationship between decimal odds, moneyline and probability is illustrated in the table below. I'll stick with decimal odds because the alternatives are either unfamiliar to me (Moneyline) or just stupid (fractional odds).

{% include odds_conversion_table.html %}

So, we have our model probabilities and (if we trust the exchange) we know the true probabilities of each event happening. Ideally, our model would identify situations the market has underestimated the chances of an event occurring (or not occurring in the case of lay bets). For example, in a simple coin toss game, imagine if you were offered $2 for every $1 wagered (plus your stake), if you guessed correctly. The implied probability is 0.333, but any valid model would return a probability of 0.5. The odds returned by our model and the Betfair exchange are compared in the table below.

{% include prob_comparison_table.html %}

Green cells illustrate opportunities to make profitable bets, according to our model (the opacity of the cell is determined by the implied difference). I've highlighted the difference between the model and Betfair in absolute terms (the relative difference may be more relevant for any trading strategy). Transparent cells indicate situations where the exchange and our model are in broad agreement. Strong colours imply that either our model is wrong or the exchange is wrong. Given the simplicity of our model, I'd lean towards the latter.

## Something's Poissony

So should we bet the house on Manchester United? Probably not ([though they did win!](https://www.theguardian.com/football/2017/may/21/manchester-united-crystal-palace-premier-league-match-report)). There's some non-statistical reasons to resist backing them. Keen football fans would notice that these matches represent the final gameweek of the season. Most teams have very little to play for, meaning that the matches are less predictable (especially when they involve unmotivated 'bigger' teams). Compounding that, Man United were set to play Ajax in the Europa Final three days later. [Man United manager, Jose Mourinho, had even confirmed that he would rest the first team, saving them for the much more important final](https://www.theguardian.com/football/2017/may/17/jose-mourinho-manchester-united-last-premier-league-game). In a similar fashion, injuries/suspensions to key players, managerial sackings would render our model inaccurate. Never underestimate the importance of domain knowledge in statistical modelling/machine learning! We could also think of improvements to the model that would [incorporate time when considering previous matches](http://opisthokonta.net/?p=890) (i.e. more recent matches should be weighted more strongly).

Statistically speaking, is a Poisson distribution even appropriate? Our model was founded on the belief that the number goals can be accurately expressed as a Poisson distribution. If that assumption is misguided, then the model outputs will be unreliable. Given a Poisson distribution with mean $$\lambda$$, then the number of events in half that time period follows a Poisson distribution with mean $$\lambda$$/2. In football terms, according to our Poisson model, there should be an equal number of goals in the first and second halves. Unfortunately, that doesn't appear to hold true.


```python
epl_1617_halves = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E0.csv")
epl_1617_halves = epl_1617_halves[['FTHG', 'FTAG', 'HTHG', 'HTAG']]
epl_1617_halves['FHgoals'] = epl_1617_halves['HTHG'] + epl_1617_halves['HTAG']
epl_1617_halves['SHgoals'] = epl_1617_halves['FTHG'] + epl_1617_halves['FTAG'] - 
                               epl_1617_halves['FHgoals']
epl_1617_halves = epl_1617_halves[['FHgoals', 'SHgoals']]
```

<div style="text-align:center" markdown="1">

![]({{ base_path }}/images/goals_per_half_python.png)

</div>



We have irrefutable evidence that violates a fundamental assumption of our model, rendering this whole post as pointless as Sunderland!!! Or we can build on our crude first attempt. Rather than a simple univariate Poisson model, we might have [more success](http://www.ajbuckeconbikesail.net/wkpapers/Airports/MVPoisson/soccer_betting.pdf) with a [bivariate Poisson distriubtion](http://www.stat-athens.aueb.gr/~karlis/Bivariate%20Poisson%20Regression.pdf). The [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution) has also been proposed as a [viable alternative](http://www.sportstradingnetwork.com/article/journal/using-the-weibull-count-distribution-for-predicting-the-results-of-football-matches/). These might be topics for future blog posts.

## Summary

We built a simple Poisson model to predict the results of English Premier League matches. Despite its inherent flaws, it recreates several features that would be a necessity for any predictive football model (home advantage, varying offensive strengths and opposition quality). In conclusion, don't wager the rent money, but it's a good starting point for more sophisticated realistic models. Thanks for reading!
