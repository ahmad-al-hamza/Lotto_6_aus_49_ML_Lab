# Statistical Analysis and Randomness Investigation

## Objective

This project investigates whether historical Lotto 6aus49 draw data contains statistical structure, temporal patterns, or predictive information that can be used to construct number selections with a higher-than-random expected number of hits.

The objective is **not** to claim that lottery outcomes can be predicted with certainty.

Instead, the project treats Lotto 6aus49 as a statistical process and asks whether historical observations provide measurable predictive information about future draws.

The central scientific question is whether the observed historical patterns are consistent with ordinary finite-sample random variation or whether they contain a reproducible signal that generalizes to unseen draws.

---

## Research Question

The primary research question is:

> Do historical Lotto 6aus49 draws provide statistically reliable information about future draws?

Under an independent random-draw hypothesis:

$$
P(X_t \mid X_{t-1}, X_{t-2}, \ldots) = P(X_t)
$$

In other words, knowledge of previous draws should not change the true probability distribution of the next draw.

The project therefore evaluates whether historical information such as frequency, recency, gaps, pairwise relationships, stability, diversity, and machine-learning features can improve predictive performance beyond a random Top-6 baseline.

The analysis does **not** attempt to prove randomness in an absolute sense. Failure to reject a random model is not equivalent to proving that the process is perfectly random.

---

## Methodology

The project uses time-ordered historical data and primarily evaluates strategies using walk-forward validation.

The basic experimental structure is:

```text
Historical data
      |
      v
Training period
      |
      v
Generate Top-6 prediction
      |
      v
Future test period
      |
      v
Compare against random baseline
```

The main baseline is the expected number of hits obtained by selecting six numbers uniformly from 1–49:

$$
E[H] = \frac{6 \times 6}{49}
     = 0.734694
$$

This represents the expected number of matching numbers per draw for a random Top-6 selection.

The project avoids ordinary random train/test splitting for the main time-dependent experiments and instead preserves chronological order.

---

## Statistical Tests

The project contains several statistical and computational evaluation procedures.

### Random baseline

Each strategy is compared with the theoretical random expectation:

$$
E[H] = \frac{36}{49}
$$

The observed difference is:

$$
D = \bar H_{model} - \bar H_{random}
$$

A positive difference means that the strategy achieved more average hits than the random baseline during the evaluated period.

### Monte Carlo random simulation

Several strategies generate thousands of random Top-6 selections and evaluate their hit distributions.

The simulations are used to estimate:

* mean random performance
* lower and upper percentile ranges
* empirical probabilities of achieving at least the observed model performance

### Hypergeometric random baseline

A vectorized implementation also uses the exact hypergeometric model for the number of hits when selecting six numbers from 49 and comparing them with six winning numbers.

### Bootstrap

The frozen OOS validation stage resamples observed performance differences with replacement to estimate a bootstrap confidence interval for the mean difference.

### Permutation/sign-flip test

The frozen validation stage also constructs a null distribution by randomly flipping the sign of observed differences and comparing the resulting mean differences with the observed mean.

### Important limitation

The project does not currently implement formal:

* Chi-square goodness-of-fit tests
* autocorrelation tests
* runs tests
* Ljung–Box tests
* entropy tests
* Markov-chain independence tests
* Bayesian inference

These should therefore not be described as performed statistical tests.

---

## Historical Pattern Analysis

### Frequency

Historical number frequency is calculated for numbers 1–49.

For number \(i\):

$$
f_i = \frac{c_i}{T}
$$

where \(c_i\) is the number of historical appearances and \(T\) is the number of draws.

The frequency can also be compared with the expected per-draw rate:

$$
p_0 = \frac{6}{49}
$$

and a relative strength can be calculated as:

$$
Strength_i = \frac{f_i}{6/49}
$$

Frequency is used as a predictive feature in several strategies.

### Hot and cold behavior

The project does not implement a formal statistical "hot/cold" hypothesis test.

However, hot/cold-like behavior is represented through:

* total frequency
* recent frequency
* recency
* gap
* recent-vs-long-term activity

These quantities are used for scoring and machine-learning features.

### Recency

Recent frequencies are calculated over windows such as 20, 50, and 100 draws.

For example:

$$
Rate_{20} = \frac{Count_{20}}{20}
$$

A momentum-style feature is also used:

$$
Momentum = Rate_{20} - Rate_{50}
$$

### Gap analysis

The project measures the number of draws since a number last appeared.

Gap information is used as a feature or score component.

The project does not assume that a large gap makes a number mathematically "due". Gap is treated as a predictive feature whose usefulness must be demonstrated by out-of-sample performance.

### Pairwise analysis

The Ensemble strategy counts historical number pairs.

For two numbers \(i,j\), an expected pair frequency is approximated from their individual counts and compared with the observed pair count.

Pairwise information contributes to the Ensemble score.

### Consecutive numbers

Consecutive-number characteristics are included in the feature-stability machine-learning experiment, including previous-draw consecutive statistics.

They are treated as predictive features rather than as a standalone statistical test of consecutive-number probabilities.

### Odd/even features

Previous-draw odd and even counts are included in the feature-stability experiment.

The project does not currently perform a formal statistical test of the full odd/even distribution.

### Sum, mean, standard deviation, and range

Previous-draw aggregate features include:

* draw sum
* draw mean
* draw standard deviation
* draw range

These features are used by the feature-stability machine-learning experiment.

They are not currently evaluated using a formal theoretical distribution test.

---

## Predictive Models

### Recency Logistic Regression

The Recency strategy constructs number-level training examples using:

* total frequency
* frequency over recent windows
* gap
* recent rates
* total rate
* momentum

A Logistic Regression model predicts the probability that each number appears in the next draw.

The highest-scoring six numbers are selected.

### Ensemble Logistic Regression

The Ensemble strategy combines:

1. historical frequency
2. conditional/recent scoring
3. Logistic Regression
4. pairwise scoring
5. regime scoring

The components are combined using fixed weights.

### Meta-Score Logistic Regression

The Meta-Score model combines:

1. ensemble score
2. recency score
3. stability score
4. diversity score
5. machine-learning score

The final Meta-Score is:

$$
MetaScore =
0.20E +
0.20R +
0.20S +
0.20D +
0.20ML
$$

The six numbers with the highest Meta-Score are selected.

### XGBoost Feature-Stability Experiment

A separate feature-stability experiment uses XGBoost to investigate whether a wider set of historical features can predict future number-level outcomes.

Features include historical draw statistics, frequency windows, gaps, temporal variables, odd/even counts, consecutive-number characteristics, and related historical features.

The key result was that training ROC-AUC was substantially higher than testing ROC-AUC, while test ROC-AUC remained close to 0.50.

This pattern is consistent with strong overfitting or the absence of a stable predictive relationship.

---

## Walk-forward Validation

Walk-forward validation is used extensively.

The training set always precedes the test set chronologically.

For example:

```text
Training: earlier historical draws
Testing:  later unseen historical draws
```

The test period is not used to construct the prediction for that fold.

This design is substantially more appropriate for temporal prediction than randomly mixing historical observations.

---

## Adaptive Meta-Models

Several adaptive meta-model iterations were developed, including V6, V7, V8, V9, and V10.

The adaptive framework learns strategy weights from previous folds and applies those weights to later folds.

The intended structure is:

$$
w_t = f(D_1,\ldots,D_{t-1})
$$

where \(D\) represents previous strategy performance.

The adaptive weighting implementation uses recency-weighted historical performance and a softmax transformation.

However, an important methodological distinction is required:

The adaptive meta-model primarily aggregates already-evaluated strategy performance differences. It should therefore not automatically be interpreted as a separately constructed predictive Top-6 ensemble unless a new combined prediction is explicitly generated before the test period.

---

## Random Baseline and Monte Carlo Simulation

Several strategies use thousands of random simulations.

The random selections are sampled without replacement from numbers 1–49.

The simulations estimate the distribution of average hits expected under random Top-6 selection.

Typical outputs include:

```text
Random expected
Random simulation mean
Random 95% range
Empirical p-value
```

These simulations are useful because apparent historical advantages can occur naturally in random processes.

A strategy that performs slightly above the theoretical expectation is not automatically evidence of predictive information.

---

## Backtesting Results

The historical experiments produced mixed fold-level results.

Some individual folds produced positive differences, occasionally substantially above the random expectation.

However, these advantages were not consistently reproduced across subsequent folds.

For example, the Meta-Score experiment produced a strong positive result in one fold but weaker or negative results elsewhere.

The overall historical Meta-Score result did not satisfy the project's own criterion for a meaningful advantage.

The Ensemble strategy also failed to demonstrate a stable advantage over random selection.

The feature-stability XGBoost experiment was particularly informative:

```text
Mean Test ROC-AUC: 0.501123
Mean Average Hits: 0.731263
Random Expected:   0.734694
Mean Difference:  -0.003431
```

A test ROC-AUC close to 0.50 indicates essentially random discrimination.

---

## Window Optimization

The project evaluated multiple historical window lengths, including:

* 250
* 500
* 750
* 1000
* 1500
* 2000

The 1000-draw window produced the highest historical mean performance in the window comparison.

However, selecting the best window after observing historical test performance creates a model-selection risk.

Therefore, the apparent advantage of the selected window should not be treated as independent evidence.

A completely unseen OOS period is required to evaluate the selected window without selection bias.

---

## Results

The final historical validation identified V10 as the highest observed model among the evaluated variants.

The reported V10 mean hit rate was approximately:

$$
0.756708
$$

with:

$$
Difference = +0.022014
$$

and an apparent relative improvement of approximately:

$$
+2.996\%
$$

However, the uncertainty interval included zero:

$$
95\%\,CI =
[-0.019573,\,+0.075614]
$$

and the permutation p-value was approximately:

$$
p=0.623438
$$

Therefore, the observed V10 advantage is not statistically robust.

Furthermore, the separate adaptive weighted V10 evaluation produced a negative weighted difference relative to the random baseline.

The overall evidence therefore does not justify claiming a genuine predictive advantage.

---

## Randomness and Independence

The project does not contain a formal autocorrelation or independence test.

Instead, independence is investigated indirectly through predictive experiments.

If previous draws contained reliable predictive information, models using historical frequency, recency, gaps, pairwise information, and historical draw characteristics should demonstrate reproducible out-of-sample improvement.

The observed results do not show such stable improvement.

The XGBoost experiment is especially relevant because it provides a broad test of historical features. Its average test ROC-AUC was approximately 0.501, very close to the random level of 0.5.

Therefore, the available predictive evidence is consistent with the hypothesis that historical draw information provides little or no useful predictive information.

This does not constitute mathematical proof of independence.

---

## Overfitting and Data Snooping

The project contains several safeguards against direct look-ahead bias:

* chronological training/testing
* walk-forward validation
* historical-only feature construction
* frozen strategy validation

However, the iterative development process creates substantial model-selection risk.

The project tested:

* multiple strategies
* multiple windows
* multiple model versions
* multiple scoring combinations
* multiple adaptive meta-models

This creates a multiple-testing problem.

The strongest historical result among many tested alternatives may occur by chance.

The difference between training and testing performance in the XGBoost experiment is also consistent with overfitting:

```text
Training ROC-AUC ≈ 0.65–0.69
Test ROC-AUC    ≈ 0.50
```

Therefore, historical performance should not be interpreted as proof of a genuine signal.

---

## Frozen Out-of-Sample Validation

The project includes a dedicated frozen OOS validation stage.

Its intended rules are:

* no model optimization
* no parameter changes
* no weight optimization
* no use of OOS performance to construct predictions

The validation stage calculates:

* mean performance difference
* bootstrap confidence interval
* permutation p-value
* above/below-random counts

This is the appropriate next stage after model development.

A true OOS dataset must remain completely unseen until the model and its parameters are frozen.

---

## Limitations

### Randomness

Lottery draws are probabilistic events. Historical deviations from expected frequencies can occur naturally.

### Finite sample size

Even thousands of draws do not guarantee that small deviations from expected frequencies represent genuine structural effects.

### Multiple comparisons

Testing many strategies, windows, features, and model versions increases the probability of observing apparently strong results by chance.

### Overfitting

Machine-learning models can learn historical noise without learning a relationship that generalizes to future draws.

### Data mining

Repeatedly modifying the model after observing previous results can turn the historical test set into an indirect training resource.

### Selection bias

Selecting the best-performing model or window after seeing historical results introduces selection bias.

### Prediction uncertainty

Average hits are not equivalent to jackpot probability.

A strategy that obtains slightly more average matching numbers does not automatically imply a proportional increase in the probability of matching all six winning numbers.

### Incomplete formal randomness testing

The project does not currently implement formal:

* Chi-square goodness-of-fit tests
* autocorrelation tests
* runs tests
* entropy tests
* Ljung–Box tests
* Markov models
* Bayesian inference

Therefore, conclusions about randomness should remain limited to the evidence actually produced by the implemented predictive experiments.

---

## Interpretation

The project found many historical patterns.

That result is expected in a finite random dataset.

The important question is whether those patterns remain useful when applied to future observations that were not used to construct the strategy.

The current evidence indicates that they do not provide a statistically robust predictive advantage.

The most important evidence is not that individual folds occasionally outperform random selection.

The important evidence is that:

1. positive fold-level results are inconsistent;
2. machine-learning test AUC remains close to 0.50;
3. ensemble methods do not produce a stable advantage;
4. adaptive V8–V10 performance does not outperform the random baseline;
5. final model-selection results have confidence intervals containing zero;
6. permutation p-values do not provide statistically significant evidence;
7. the best observed historical model is not statistically robust.

---

## Conclusion

The project does not prove that Lotto 6aus49 is mathematically random.

Instead, the analysis provides insufficient evidence to reject a model in which historical draw results do not provide reliable predictive information about future draws.

The current evidence therefore supports the conservative conclusion:

> **The analysis found no statistically robust evidence that historical Lotto 6aus49 draw results provide predictive information that consistently improves Top-6 selection beyond the random baseline.**

The appropriate scientific next step is not another adaptive model iteration.

The appropriate next step is:

```text
Freeze the selected strategy
        ↓
Generate predictions without OOS information
        ↓
Collect genuinely unseen draws
        ↓
Evaluate the frozen strategy
        ↓
Only consider further development if a robust OOS signal appears
```

No individual number combination should be interpreted as guaranteed to win.

The purpose of this project is statistical investigation of predictability, dependence, and potential signal—not guaranteed lottery prediction.
