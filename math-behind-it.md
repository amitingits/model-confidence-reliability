
# Model Confidence vs Data Quality

## Laws, Mathematics, and Project Architecture

---

## 1. The Core Problem

Modern machine learning models do not just make predictions — they also report **confidence** in those predictions.
In many real systems (healthcare, finance, security), decisions are made not only based on *what* the model predicts, but **how confident it claims to be**.
However, a critical and often untested assumption is:

> *If a model is confident, it is likely correct.*

This assumption is **not guaranteed**, especially when data quality degrades in real-world environments.

The core question studied is:

![Core Question](https://latex.codecogs.com/svg.image?\hat{P}(\text{correct}\mid&space;x)\approx&space;P(\text{correct}\mid&space;x)) still hold when data quality degrades?

**where:**

* ![\hat{P}](https://latex.codecogs.com/svg.image?\hat{P}) is the model’s predicted probability
* ![P](https://latex.codecogs.com/svg.image?P) is the true empirical probability

This equality is **assumed**, but rarely **tested under degraded data**.

> **This project studies how the relationship between model confidence and actual correctness degrades as data quality deteriorates in controlled, measurable ways.**

It is not about building better models.
It is about **measuring trustworthiness**.

---

## 2. Mathematical Setup (Foundation)

### Dataset
The study is built upon a dataset defined as:

![Dataset](https://latex.codecogs.com/svg.image?\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n,&space;\qquad&space;x_i\in\mathbb{R}^d,&space;\qquad&space;y_i\in\{1,\dots,K\})

**where:**

* ![\mathcal{D}](https://latex.codecogs.com/svg.image?\mathcal{D}) is the dataset
* ![x\_i](https://latex.codecogs.com/svg.image?x_i) is the feature vector
* ![y\_i](https://latex.codecogs.com/svg.image?y_i) is the ground-truth label

---

### Trained Probabilistic Classifier
We utilize a probabilistic classifier $f_\theta$ that maps input features to a probability simplex:

![Classifier](https://latex.codecogs.com/svg.image?f_\theta:\mathbb{R}^d\rightarrow\Delta^{K-1})


![Probabilities](https://latex.codecogs.com/svg.image?f_\theta\(x_i\)=\hat{p}_i=\(\hat{p}_{i1},\dots,\hat{p}_{iK}\))

**where:**

* ![\theta](https://latex.codecogs.com/svg.image?\theta) are model parameters
* ![\hat{p}\_{ik}](https://latex.codecogs.com/svg.image?\hat{p}_{ik}) is predicted probability for class ![k](https://latex.codecogs.com/svg.image?k)

---

## 3. The Three Fundamental “Laws” of the Project

The entire project is governed by three strict principles.
Every architectural decision enforces them.

### **Law 1: Confidence Is a Quantitative Claim**

A model’s confidence is **not a feeling or heuristic**.
It is a **numerical probability**.

If a model outputs:
* “Class A with probability 0.92”
Then it is implicitly claiming:

> *Out of many similar cases, I should be correct roughly 92% of the time.*

This project treats confidence as a **statistical promise**, not a cosmetic number.
To evaluate this claim, we define the following mathematical objects:

**1. The Prediction ($\hat{y}_i$)**
The model selects the class with the highest predicted probability:

![Prediction](https://latex.codecogs.com/svg.image?\hat{y}_i=\arg\max_k\hat{p}_{ik})

**2. The Confidence Score ($c_i$)**
The confidence associated with that prediction is the magnitude of the maximum predicted probability:

$$c_i = \max_k \hat{p}_{ik}$$

**3. The Interpretation of Reliability**
This project treats confidence as a **statistical promise**, not a cosmetic number. A model is considered perfectly reliable if the empirical probability of being correct matches the claimed confidence:

$$P(\hat{y}_i = y_i \mid c_i \approx 0.87) \approx 0.87$$


Confidence is a **probabilistic promise**, not a heuristic.

---

## 4. Law 2: Correctness Is Empirical
Correctness is defined **only** by comparison with ground truth.

A prediction is:

* Correct → 1
* Incorrect → 0

There is no partial credit, no interpretation.

This allows confidence (a probabilistic belief) to be compared against correctness (a binary outcome) **mathematically**.

**1. The Correctness Indicator ($z_i$)**
We define a binary variable to represent the accuracy of a single prediction:

$$z_i = \begin{cases} 1 & \text{if } \hat{y}_i = y_i \\ 0 & \text{otherwise} \end{cases}$$

**2. The Statistical Nature of Correctness**
Correctness is treated as a **Bernoulli** random variable, allowing confidence (a probabilistic belief) to be compared against correctness (a binary outcome) mathematically:

$$z_i \sim \mathrm{Bernoulli}(P(\text{correct}))$$

---

## 5. Law 3 — Reliability Under Degradation
Reliability is not a static property; it is a function of data quality. This project assumes that a model reliable only on clean data is not truly reliable. 

We define a **Degradation Operator** $\mathcal{T}_d$ that transforms clean data $(X, Y)$ into degraded data $(X^{(d)}, Y^{(d)})$ based on a severity parameter $d$:

$$\mathcal{T}_d: (X, Y) \rightarrow (X^{(d)}, Y^{(d)})$$

**Where:**
* **$\mathcal{T}_0$**: is the identity operator (clean data).
* **$d \uparrow$**: implies increasing degradation severity.
* **$f_\theta$**: the model remains frozen $\forall d$, simulating real-world silent degradation.

Therefore, reliability is tested **only under controlled degradation**.

---
## 6. Degradation & Noise Layer

The project simulates three primary modes of data failure to test when the confidence-accuracy assumption breaks.

### Missingness Models (Information Loss)
1. **MCAR (Missing Completely at Random):** Data disappears unpredictably.

   $$P(x_{ij}^{(d)} = \text{NaN}) = p_d$$

2. **MAR (Missing at Random):** Data disappears based on observed context.

   $$P(x_{ij}^{(d)} = \text{NaN} \mid x_{ik}) = \sigma(\alpha_d x_{ik})$$

3. **MNAR (Missing Not at Random):** Extreme or sensitive values disappear, often the most dangerous mode.

   ![MNAR](https://latex.codecogs.com/svg.image?x_{ij}^{(d)}%3D%5Cbegin%7Bcases%7D%5Ctext%7BNaN%7D%20%26%20x_{ij}%5Cge%20Q_{1-d}(x_j)%20%5C%5C%20x_{ij}%20%26%20%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)


### Noise & Bias Models
* **Feature Noise (Measurement Error):** Simulates sensor inaccuracy.

  $$x_i^{(d)} = x_i + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_d^2 I)$$

* **Label Noise:** Simulates human entry errors or mislabeling.

  ![Label Noise](https://latex.codecogs.com/svg.image?y_i^{(d)}%3D%5Cbegin%7Bcases%7D%5Ctext%7BUniform%7D(%5C%7B1%2C%5Cdots%2CK%5C%7D)%26%5Ctext%7Bw.p.%20%7D%5Ceta_d%5C%5Cy_i%26%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)

* **Structural Bias:** Asymmetric reliability failure where some groups are affected more than others.

  $$P(\mathcal{T}_d(x) \mid y=k_1) \neq P(\mathcal{T}_d(x) \mid y=k_2)$$


---

## 7. Model Layer: Why Models Are Frozen

Models are trained **once** on clean data and then frozen forever.

This is intentional.

Why?

Because in real deployment:

* Models are rarely retrained immediately
* Data quality often degrades silently
* Confidence outputs are still trusted

This layer answers:

> *What happens when a trusted model is exposed to untrusted data?*

---

## 8. Measurement Layer

To scientifically quantify reliability, the project employs three complementary metrics that track how the model's internal beliefs deviate from external reality.

### Accuracy (Empirical Performance)
Measures the proportion of correct predictions at a specific degradation level $d$:

$$\mathrm{Acc}(d) = \frac{1}{n} \sum_{i=1}^n z_i^{(d)}$$

### Expected Calibration Error (ECE)
Quantifies how far the model's predicted confidence deviates from its actual accuracy across $M$ probability bins:

$$\mathrm{ECE}(d) = \sum_{m=1}^M \frac{|B_m|}{n} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|$$



### Confidence–Correctness Gap ($\Delta$)
The central failure signal of this study. It measures the systematic difference between the expected confidence and the expected correctness:

$$\Delta(d) = \mathbb{E}[c_i^{(d)}] - \mathbb{E}[z_i^{(d)}]$$

**Interpretation of the Gap:**
* **$\Delta(d) > 0$**: **Overconfidence** — The model believes it is more accurate than it actually is (Safety Risk).
* **$\Delta(d) = 0$**: **Perfectly Reliable** — The model is "honest" about its uncertainty.
* **$\Delta(d) < 0$**: **Underconfidence** — The model is more accurate than it claims to be.

---

## 9. Statistical Layer

Single numbers are meaningless.

Therefore:

* Every metric is computed across multiple random seeds
* Uncertainty is estimated using resampling
* Trends are tested statistically, not visually

This ensures that conclusions are **not artifacts of randomness**.

To ensure the results are not artifacts of randomness, we apply rigorous statistical validation.

### Bootstrap Confidence Intervals (CI)
We estimate the uncertainty of our metrics by resampling the data and calculating the 95% confidence interval using the 2.5th and 97.5th percentiles of the bootstrap distribution:

![Bootstrap](https://latex.codecogs.com/svg.image?\mathrm{CI}_{95\%}=[Q_{0.025},Q_{0.975}])


### Monotonicity of Failure
We measure the relationship between the severity of degradation ($d$) and the increase in Calibration Error using the Spearman rank correlation coefficient:

$$\rho = \mathrm{Spearman}(d, \mathrm{ECE}(d))$$

A high $\rho$ indicates that as data quality drops, the model's reliability fails in a predictable, systematic way.

## 10. Architecture as Law Enforcement

| Folder        | Mathematical Object                                                              |
| ------------- | -------------------------------------------------------------------------------- |
| `data/`       | ![\mathcal{D}](https://latex.codecogs.com/svg.image?\mathcal{D})                 |
| `corruption/` | ![\mathcal{T}\_d](https://latex.codecogs.com/svg.image?\mathcal{T}_d)            |
| `models/`     | ![f\_\theta](https://latex.codecogs.com/svg.image?f_\theta)                      |
| `metrics/`    | ![\mathrm{ECE},\Delta](https://latex.codecogs.com/svg.image?\mathrm{ECE},\Delta) |
| `analysis/`   | statistical tests                                                                |

---

## 11. Formal Result

The project demonstrates the existence of **Silent Failure**:

$$\exists d : \mathbb{E}[c_i^{(d)}] \gg \mathbb{E}[z_i^{(d)}]$$

This proves that as data quality ($d$) degrades, a model can remain highly confident while its actual correctness collapses.

A model can be **confidently wrong**.

---

## 12. One-Line Summary

At its core, this project defines and measures **Reliability** as a function of data degradation ($d$):

$$\mathrm{Reliability}(d) = \left| \mathbb{E}[c_i^{(d)}] - \mathbb{E}[z_i^{(d)}] \right|$$

We study this relationship as $d \uparrow$ (severity increases) to determine the exact point where a model's statistical promise stops being meaningful.

> **Project Mission:** To measure when and why machine learning models stop being honest about their own uncertainty as data quality deteriorates.

## 13. Why This Project Matters

Because in real systems:

* Overconfident models cause harm
* Silent data degradation is common
* Confidence is trusted more than it should be

This project provides a **systematic, mathematical way to test that trust**.
