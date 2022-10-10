functions {
    real likehood_ql2(
        real alpha,
        real beta,
        int[] actions,
        int[] rewards,
        int trialCount
    ) {
        row_vector[2] Q = [0.5, 0.5];
        real choiceProbabilities[trialCount];
        row_vector[2] p;
        real delta;

        for (trial in 1:trialCount) {

            // compute choice probabilities
            p = softmax(Q' * beta)';  //softmax wants vector so transposing

            // add choice probability for actual choice
            choiceProbabilities[trial] = p[actions[trial]];

            // update values
            delta = rewards[trial] - Q[actions[trial]];
            Q[actions[trial]] += alpha * delta;
        }

        // return log-likelihood
        return sum(log(choiceProbabilities));
    }
}
data {
    int trial_count;
    int action_seq[trial_count];
    int reward_seq[trial_count];
}
parameters {
    real<lower=0, upper=1> alpha;
    real<lower=0> beta;
}
model {
    alpha ~ uniform(0, 1);
    beta ~ uniform(1, 40);
    target += likehood_ql2(alpha, beta, action_seq, reward_seq, trial_count);
}
generated quantities {
    real log_like;
    log_like = likehood_ql2(alpha, beta, action_seq, reward_seq, trial_count);
}
