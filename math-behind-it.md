
# Model Confidence vs Data Quality

## Laws, Mathematics, and Project Architecture

---

## 1. What Problem This Project Solves (Formal)

A deployed machine learning system produces **predictions** and **confidence scores**.
The core question studied is:

![Core Question](https://latex.codecogs.com/svg.image?\hat{P}(\text{correct}\mid&space;x)\approx&space;P(\text{correct}\mid&space;x)) still hold when data quality degrades?

**where:**

* ![\hat{P}](https://latex.codecogs.com/svg.image?\hat{P}) is the model’s predicted probability
* ![P](https://latex.codecogs.com/svg.image?P) is the true empirical probability

This equality is **assumed**, but rarely **tested under degraded data**.

---

## 2. Mathematical Setup (Foundation)

### Dataset

![Dataset](https://latex.codecogs.com/svg.image?\mathcal{D}=\{(x_i,y_i)\}_{i=1}^n,&space;\qquad&space;x_i\in\mathbb{R}^d,&space;\qquad&space;y_i\in\{1,\dots,K\})

**where:**

* ![\mathcal{D}](https://latex.codecogs.com/svg.image?\mathcal{D}) is the dataset
* ![x\_i](https://latex.codecogs.com/svg.image?x_i) is the feature vector
* ![y\_i](https://latex.codecogs.com/svg.image?y_i) is the ground-truth label

---

### Trained Probabilistic Classifier


![Classifier](https://latex.codecogs.com/svg.image?f_\theta:\mathbb{R}^d\rightarrow\Delta^{K-1})


![Probabilities](https://latex.codecogs.com/svg.image?f_\theta\(x_i\)=\hat{p}_i=\(\hat{p}_{i1},\dots,\hat{p}_{iK}\))

**where:**

* ![\theta](https://latex.codecogs.com/svg.image?\theta) are model parameters
* ![\hat{p}\_{ik}](https://latex.codecogs.com/svg.image?\hat{p}_{ik}) is predicted probability for class ![k](https://latex.codecogs.com/svg.image?k)

---

## 3. Law 1 — Confidence Is a Statistical Claim

### Prediction

![Prediction](https://latex.codecogs.com/svg.image?\hat{y}_i=\arg\max_k\hat{p}_{ik})

### Confidence

![Confidence](https://latex.codecogs.com/svg.image?c_i=\max_k\hat{p}_{ik})

**Interpretation**

![Formula](https://latex.codecogs.com/svg.image?P(\hat{y}_i=y_i|c_i\approx0.87)\approx0.87)

Confidence is a **probabilistic promise**, not a heuristic.

---

## 4. Law 2 — Correctness Is Empirical

Correctness is not **inferred**, it is **measured**:


![Correctness](https://latex.codecogs.com/svg.image?z_i%3D%5Cbegin%7Bcases%7D1%26%5Ctext%7B%20if%20%7D%5Chat%7By%7D_i%3Dy_i%5C%5C0%26%5Ctext%7B%20otherwise%7D%5Cend%7Bcases%7D)

Correctness is a **Bernoulli** random variable:

![Bernoulli](https://latex.codecogs.com/svg.image?z_i\sim\mathrm{Bernoulli}(P(\text{correct})))

---

## 5. Law 3 — Reliability Under Degradation

![Degradation Operator](https://latex.codecogs.com/svg.image?\mathcal{T}_d:\(X,Y\)\rightarrow\(X^{\(d\)},Y^{\(d\)}\))

**where:**

* ![\mathcal{T}\_0](https://latex.codecogs.com/svg.image?\mathcal{T}_0) is the identity operator
* ![d\uparrow](https://latex.codecogs.com/svg.image?d\uparrow) implies increasing degradation
* ![f\_\theta](https://latex.codecogs.com/svg.image?f_\theta) is fixed ∀ ![d](https://latex.codecogs.com/svg.image?d)

---

## 6. Degradation Models

### MCAR

![MCAR](https://latex.codecogs.com/svg.image?P(x_{ij}^{(d)}=\text{NaN})=p_d)

### MAR

![MAR](https://latex.codecogs.com/svg.image?P(x_{ij}^{(d)}=\text{NaN}\mid&space;x_{ik})=\sigma(\alpha_d&space;x_{ik}))

### MNAR

![MNAR](https://latex.codecogs.com/svg.image?x_{ij}^{(d)}%3D%5Cbegin%7Bcases%7D%5Ctext%7BNaN%7D%20%26%20x_{ij}%5Cge%20Q_{1-d}(x_j)%20%5C%5C%20x_{ij}%20%26%20%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)
---

## 7. Noise Models

### Feature Noise

![Feature Noise](https://latex.codecogs.com/svg.image?x_i^{(d)}=x_i+\epsilon,\;\epsilon\sim\mathcal{N}(0,\sigma_d^2I))


### Label Noise

![Label Noise](https://latex.codecogs.com/svg.image?y_i^{(d)}%3D%5Cbegin%7Bcases%7D%5Ctext%7BUniform%7D(%5C%7B1%2C%5Cdots%2CK%5C%7D)%26%5Ctext%7Bw.p.%20%7D%5Ceta_d%5C%5Cy_i%26%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D)

---

## 8. Bias Models

![Bias](https://latex.codecogs.com/svg.image?P(%5Cmathcal%7BT%7D_d(x)%5Cmid%20y%3Dk_1)%5Cneq%20P(%5Cmathcal%7BT%7D_d(x)%5Cmid%20y%3Dk_2))

Bias causes **asymmetric reliability failure**.

---

## 9. Measurement Layer

### Accuracy

![Accuracy](https://latex.codecogs.com/svg.image?\mathrm{Acc}(d)=\frac{1}{n}\sum_{i=1}^nz_i^{(d)})

---

### Expected Calibration Error

![ECE](https://latex.codecogs.com/svg.image?\mathrm{ECE}(d)=\sum_{m=1}^M\frac{|B_m|}{n}\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|)

---

### Confidence–Correctness Gap

![Gap](https://latex.codecogs.com/svg.image?\Delta(d)=\mathbb{E}[c_i^{(d)}]-\mathbb{E}[z_i^{(d)}])

**where:**

* ![\Delta(d)>0](https://latex.codecogs.com/svg.image?\Delta\(d\)>0) → overconfidence
* ![\Delta(d)=0](https://latex.codecogs.com/svg.image?\Delta\(d\)=0) → reliable
* ![\Delta(d)<0](https://latex.codecogs.com/svg.image?\Delta\(d\)<0) → underconfidence

---

## 10. Statistical Laws

### Bootstrap CI

![Bootstrap](https://latex.codecogs.com/svg.image?\mathrm{CI}_{95\%}=[Q_{0.025},Q_{0.975}])


### Monotonicity

![Spearman](https://latex.codecogs.com/svg.image?\rho=\mathrm{Spearman}(d,\mathrm{ECE}(d)))

---

## 11. Architecture as Law Enforcement

| Folder        | Mathematical Object                                                              |
| ------------- | -------------------------------------------------------------------------------- |
| `data/`       | ![\mathcal{D}](https://latex.codecogs.com/svg.image?\mathcal{D})                 |
| `corruption/` | ![\mathcal{T}\_d](https://latex.codecogs.com/svg.image?\mathcal{T}_d)            |
| `models/`     | ![f\_\theta](https://latex.codecogs.com/svg.image?f_\theta)                      |
| `metrics/`    | ![\mathrm{ECE},\Delta](https://latex.codecogs.com/svg.image?\mathrm{ECE},\Delta) |
| `analysis/`   | statistical tests                                                                |

---

## 12. Formal Result

![Failure](https://latex.codecogs.com/svg.image?%5Cexists%20d%3A%5Cmathbb%7BE%7D%5Bc_i%5E%7B(d)%7D%5D%5Cgg%5Cmathbb%7BE%7D%5Bz_i%5E%7B(d)%7D%5D)

A model can be **confidently wrong**.

---

## 13. One-Line Summary

![Reliability](https://latex.codecogs.com/svg.image?\mathrm{Reliability}(d)=\left|\mathbb{E}[c_i^{(d)}]-\mathbb{E}[z_i^{(d)}]\right|)

Studied as ![d\uparrow](https://latex.codecogs.com/svg.image?d\uparrow).

