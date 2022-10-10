data {
    int trial_count;
    int action_seq[trial_count];
    int reward_seq[trial_count];
}
parameters {
    real<lower=0, upper=1> bias;
}
model {
    // random bias algorithm
    bias ~ uniform(0, 1);
    vector[2] choice_values = [bias, 1-bias]';
    
    for (trial in 1:trial_count) {
        
        // compute choice/action probabilities
        action_seq[trial] ~ categorical_logit(choice_values);
    }
}
generated quantities {
    real log_like;
    vector[2] choice_values = [bias, 1-bias]';

    log_like = 0;
    
    for (trial in 1:trial_count) {
        log_like += categorical_logit_lpmf(action_seq[trial] | choice_values);
    }
}
