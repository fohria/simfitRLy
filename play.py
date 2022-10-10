# %% markdown
# ## quick examples
#
# import may give warning from tqdm that you don't have proper widget support. doesn't matter unless you want fancy progress bars.

# %%
import simfitrly as sf
import pandas as pd
import seaborn as sns

# setup so plots look reasonably non-ugly
sns.set(
    rc={
        "figure.dpi": 200,
        "figure.figsize": (10, 6),
        "axes.facecolor": "#f0f0f0",
        "grid.color": "#ccc",
    }
)
# %% markdown

# ### create a task
#
# create a task like below. you can pick between:
#
# - Bandit (standard 2 arm bandit with rewards 0/1)
# - ReversalBandit (2 arm bandit that will switch best arm)
# - WorthyBandit (2 decks of cards that can have rewards 1-10)
# - Shapetask (predict the next shape based on current/previous shapes)
#
# please see thesis document for more info on each task.

# %%
# create a new bandit where we do 100 arm pulls - trial count
# left arm has 0.2 probability of reward, right arm has 0.7 reward probability
task = sf.tasks.Bandit(arm1=0.2, arm2=0.7, trial_count=100)

# %% markdown
# ### create an agent
#
# there are multiple agents available, not all of them can play all tasks. two simple agents are:
#
# - QL (Q-learning)
# - RandomBias (picks an option at random, with a possible bias for one option)
#
# check the `simulate/agents` folder for more agents, and the thesis document for more elaborate descriptions

# %%
agent = sf.agents.QL()

# %% markdown
# each agent has required parameter values, so above will give you an error if these are not provided, like below.
#
# the `QL` agent also has an optional `gamma` parameter for temporal discount, which is useful if multiple future steps are needed before a reward. for bandit tasks we usually get rewards immediately so only two parameters are necessary.

# %%
agent = sf.agents.QL(alpha=0.3, beta=5)

# %% markdown
# ### play the task
#
# the agents have `play` functions that take the task object as input. if the agent can't play the particular task it should give you an error, otherwise it returns a dictionary with three keys: `actions`, `rewards`, `stimuli`. the values of each key is a numpy array with the actions taken on each trial, the rewards received, and stimuli encountered. The stimuli array will be filled with `None` if the task does not use stimuli, like most Bandit tasks.

# %%
agent.play(task)

# %% markdown
# ### save results and create a tidy dataframe
#
# [tidy data](http://vita.had.co.nz/papers/tidy-data.html) is nice for later plotting of the results. we have a convenience function `tidy_dataframe` for this purpose. a nice future improvement would be to send the results dictionary into this function instead of separating actions, rewards and stimuli as of now.

# %%
results = agent.play(task)
results = sf.utils.tidy_dataframe(
    results["actions"],
    results["rewards"],
    results["stimuli"],
    agent,
    task)
results

# %% markdown
# ### score the task
#
# each task has a built-in scoring function that calculates, well, relevant scores we can use to analyse the results.
#
# the current code may give a future warning from pandas, i've yet to look into what it entails. for now we can ignore it.

# %%
scores = task.score_experiment(results)
scores

# %% markdown
# ### visualise result
#
# the score function above defines a correct choice as picking the arm with the highest reward probability. we can do a simple plot showing how the QL agent learns what arm is the best

# %%
sns.lineplot(data=scores, x="trial", y="correct")

# %% markdown
# ### fit model to data and recover parameters
#
# using the data from above, we can fit the `QL` and `RandomBias` models (when fitting we use the term model instead of agent which we use when simulating) and see which one is more likely to have generated said data.

# %%
ql2_model = sf.models.QL2()

# %% markdown
# the behavioural data we need are actions (choices) and rewards. we have those in our tidy dataframe from before.
#
# however, we use numba, a library that compiles python into C code, to speed up our fitting functions. numba is quite picky about types, and will complain if we give it pandas series. so we convert from pandas to numpy before using the fitting function.
#
# what happens in the `fit_bandit_mle` function is that based on just the actions and rewards data, we try to "recover" the true parameter values that we used when generating the data (or "playing" the task).

# %%
actions = results.action.to_numpy()
rewards = results.reward.to_numpy()
fit_result = ql2_model.fit_bandit_mle(actions, rewards)
fit_result

# %%
# the returned object is the output of scipy's `minimize` function, where `fun` in our case is the returned log likelihood value and `x` is the estimated `alpha` and `beta` parameters (in that order).
#
# depending on the specific data generated above, these estimated parameters may or may not be close to the true parameters. my thesis (see link in the readme) discusses in depth why this kind of parameter value recovery is such a difficult problem.
#
# nevertheless, a good practice here is to visualise the distance in a plot. the true parameter combination is marked with an orange cross (because x marks the spot) and the estimated parameter values are represented by the blue circle.

# %%
plot_data = {
    "alpha": [agent.alpha, fit_result.x[0]],
    "beta": [agent.beta, fit_result.x[1]],
    "label": ["true", "estimated"]
}

df = pd.DataFrame.from_dict(plot_data)
fig = sns.scatterplot(data=df, x="alpha", y="beta", style="label", hue="label")

# %% markdown
# ### fit multiple models and select best fit
#
# in the above example we know what model (agent) generated the data. but often we don't, and would like to fit multiple models and then compare how well they fit the data.
#
# to exemplify this we will now fit the `RandomBias` model to the same data, and then compare its fit to the ql2 model fit.

# %%
randbias_model = sf.models.RandomBias()
randbias_fit_result = randbias_model.fit_bandit_mle(actions, rewards)
randbias_fit_result

# %% markdown
# the `RandomBias` model has one parameter, the bias for arm1 ("left") over arm2 ("right"). a bias value close to 0 means a bias towards left, close to 1 means a bias towards the right, and if the bias value is 0.5 it means it will pick each arm with equal probability.
#
# when we do not know what model has generated some data, a basic way of checking the fit result is to use the recovered parameter value and simulate the behaviour.

# %%
random_agent = sf.agents.RandomBias(bias1=randbias_fit_result.x[0])
random_play_result = random_agent.play(task)
random_results = sf.utils.tidy_dataframe(
    random_play_result["actions"],
    random_play_result["rewards"],
    random_play_result["stimuli"],
    random_agent,
    task)
random_scores = task.score_experiment(random_results)
sns.lineplot(data=random_scores, x="trial", y="correct")

# %% markdown
# does this behaviour look similar to the previous plot from before? most likely not, but because all these algorithms are inherently random, it's not impossible the behaviour is similar.
#
# ### compare likelihoods
#
# the more analytical way of comparing our fits is to compare the log likelihoods. we have used `mle` functions which is short for "maximum likelihood". the observant reader will note we mentioned a `minimize` function above. what this means is that the `fun` value in our fit result objects should switch signs.
#
# we collect our aquired simulation and fit data like so:

# %%
# our tidy_mle function is built to accept multiple rows of fit results for data with multiple subjects
# thus, we here create a nested array with just one row - one subject
subject_id = 0

ql_fitdata = [[subject_id, -fit_result.fun, fit_result.x[0], fit_result.x[1]]]
ql_fitdata = sf.tidy_mle(fitdata=ql_fitdata, simdata=results, model=ql2_model)

random_fitdata = [[subject_id, -randbias_fit_result.fun, randbias_fit_result.x[0]]]
random_fitdata = sf.tidy_mle(fitdata=random_fitdata, simdata=results, model=randbias_model)

fitdata = pd.concat([ql_fitdata, random_fitdata])
fitdata

# %% markdown
# comparing the likelihood values directly is possible, but not ideal. the more parameters a model has, the easier it is that it may overfit. one way of compensating for this is to use the Bayesian Information Criterion (BIC) which takes the number of parameters into account when calculating a new comparison value based on the likelihood. there are many alternatives to the BIC, and this is also subject to discussion in my phd thesis linked in the readme file.
#
# regardless, we can now compare our two models with the BIC value. the best fitting model is the one with the lowest such BIC value, so if we only print that column from our combined `fitdata` dataframe, we get:

# %%
fitdata[["model_name","bic"]]

# %% markdown
# was the correct model selected as the best fitting one?
#
# ### to be continued
#
# this is just a small taste of what simfitRLy can do. there will be more examples in the future. for now, my next steps are to fix up the structure of the fit module and then create a pip/conda package, as well as implement tests and what nots to mayhaps make this into a "proper" python package.
