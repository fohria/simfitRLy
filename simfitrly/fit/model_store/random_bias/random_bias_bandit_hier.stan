data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=-1, upper=2> choice[N, T];
  real outcome[N, T];  // no lower and upper bounds
}
transformed data {
  // vector[2] initV;  // initial values for EV
  // initV = rep_vector(0.0, 2);
}
parameters {
// Declare all parameters as vectors for vectorizing
  // Hyper(group)-parameters
  real mu_pr;
  real<lower=0> sigma;

  // Subject-level raw parameters (for Matt trick)
  vector[N] bias_pr;    // bias
}
transformed parameters {
  // subject-level parameters
  vector<lower=0, upper=1>[N] bias;

  for (i in 1:N) {
    bias[i]   = Phi_approx(mu_pr  + sigma  * bias_pr[i]);
  }
}
model {
  // Hyperparameters
  mu_pr  ~ normal(0, 1);
  sigma ~ normal(0, 0.2);

  // individual parameters
  bias_pr   ~ normal(0, 1);

  // subject loop and trial loop
  for (i in 1:N) {
    vector[2] choice_probs;  // arm choice probabilities

    choice_probs = [bias[i], 1-bias[i]]';

    for (t in 1:(Tsubj[i])) {
      // compute action probabilities
      choice[i, t] ~ categorical_logit(choice_probs);
    }
  }
}
generated quantities {
  // For group level parameters
  real<lower=0, upper=1> mu_bias;

  // For log likelihood calculation
  real log_lik[N];

  // For posterior predictive check
  real y_pred[N, T];

  // Set all posterior predictions to 0 (avoids NULL values)
  // kept this from delta model, but it's set to -1 not 0?
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }

  mu_bias   = Phi_approx(mu_pr);

  { // local section, this saves time and space
    for (i in 1:N) {
      vector[2] choice_probs;  // arm choice probabilities

      choice_probs = [bias[i], 1-bias[i]]';

      log_lik[i] = 0;

      for (t in 1:(Tsubj[i])) {
        // compute log likelihood of current trial
        log_lik[i] += categorical_logit_lpmf(choice[i, t] | choice_probs);

        // generate posterior prediction for current trial
        y_pred[i, t] = categorical_rng(softmax(choice_probs));
      }
    }
  }
}
