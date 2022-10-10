# some notes on fit functions

hello again its may22 . ouch. we want to do with fit folder similar to what we did with simulate folder, so we can do `fit.ql2` for example. 

---
hello again. still in feb22. we have combined a bunch of thoughts below into somewhat okay structure. perhaps.

so we define all our models in `models.py` as dataclasses.

in each such dataclass, we have internal variables holding all the info for that model, like what tasks it can do and where the model files/likelihood functions for each such model/task combo are located. we also have the names of the model's parameters and its bounds for mle fitting for example. in stan we have to define such bounds in the model itself.

each model dataclass then also has functions for the tasks it can fit. so for QL2 model for example, there's a `fit_bandit` function, which takes as input `actions, rewards` as those are the data needed to fit. there's also an optional `method` argument so we can select what fitting method to use. this function uses the internal variables of the dataclass to get all the info it needs. the function itself is actually defined in `mle.py` for mle fitting and in the future, `bayes.py` for stan models.

hopefully this will be easy enough to use. as an example, this is how we can quickly simfit ql2 playing bandit and ql2 model on that data:

```python
from simulate import agents
from simulate import tasks
from fit import models

agent = agents.QL(0.4, 5)
task = tasks.Bandit(0.2, 0.8, 100)
ql2 = models.QL2()

actions, rewards = agent.play_bandit(task)
ql2.fit_bandit(actions, rewards)
```

---
hello this is me from feb22, calling out future me.

`fit_parameters` idk if that's needed, but if we keep it, why not use it better?

what i mean is, in `fit/shapetask.py` there's a `fit_mcmc` function that uses a dictionary called `model_parameters`.

that should be in fitparameters! or some other file here, depending on what we decide. but having them in there is .. inconvenient, at best.

## old notes below

so i was thinking we could, when we got all working again, indeed have a folder called shapetask here but it would contain the general python functions for fitting, the actual likelihoods/stan models would still be kept in the models folder.

so, it'll be a bit weird and contrived to get single subject mle fit functions to be shared for all methods, but advantage is we can then control bounds and start guesses in a single location and potentially later add those as optional to send with fitparameters. because, as an example, i just discovered that bql mle function has 1-30 as beta start guess interval while seql has 1-20. such things we would more easily see if we move those limits "up" so to speak, although it's really difficult to figure out how to do that.

## new suggestion perhaps
we could have a file `fit/fitters.py` wherein we have `fit_shapetask` and `fit_bandit` . the `fit_shapetask` one is the same function that is currently in `fit/shapetask.py` and called just `fit`.
that would allow us to have a `fit/shapetask` folder with all the functions currently in `fit/shapetask.py` sorted into `fit/shapetask/mle.py` and `fit/shapetask/mcmc.py`
