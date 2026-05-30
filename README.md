# Credit Scoring Business Understanding

## 1. Basel II and the Need for an Interpretable and Well-Documented Model

The Basel II Accord emphasizes risk-sensitive capital allocation by requiring financial institutions to measure, monitor, and manage credit risk accurately. Under the Internal Ratings-Based (IRB) approach, banks are responsible for developing their own models to estimate key risk parameters such as Probability of Default (PD). Since these models directly influence regulatory capital requirements, institutions must demonstrate their reliability, accuracy, and governance to regulators.

This regulatory environment creates a strong need for interpretable and well-documented credit scoring models for several reasons:

* **Regulatory Compliance:** Supervisors must be able to understand how model outputs are generated and verify that risk estimates are reasonable.
* **Minimum Capital Requirements (Pillar 1):**Defines the rules for calculating the minimum capital a bank must hold to cover credit, operational, and market risks
* **Supervisory Review (Pillar 2):** Regulators evaluate internal risk assessment processes and require sufficient documentation to identify, understand, and manage model risk.
* **Market Discipline (Pillar 3):** Transparency requirements encourage institutions to disclose information about their risk management practices and risk profiles.
* **Model Risk Management:** Interpretable models help institutions detect weaknesses, validate assumptions, and ensure that models continue to perform during economic downturns.
* **Auditability and Governance:** Basel-compliant model development requires a complete audit trail, including feature selection, model assumptions, validation procedures, back-testing results, and ongoing monitoring.
* **Consistency in Capital Requirements:** Well-documented models reduce variability caused by inconsistent modeling practices and improve comparability across institutions.

As a result, Basel II encourages the use of models that are not only predictive but also explainable, transparent, and supported by comprehensive documentation.

---

## 2. Why a Proxy Variable Is Necessary and the Risks It Introduces

In many real-world credit scoring projects, a direct measure of loan default is unavailable. This is especially common when working with alternative data sources such as payment platforms, e-commerce transactions, or mobile money providers. Since supervised machine learning requires a target variable, a proxy variable must be created to represent credit risk.

### Why a Proxy Variable Is Necessary

A proxy variable serves as a substitute for actual default behavior when default data is unavailable.

Common reasons include:

* **Lack of Loan Performance Data:** Alternative data providers often do not have access to borrowers' repayment histories with financial institutions.
* **Requirement for Supervised Learning:** Machine learning models need labeled outcomes to learn relationships between customer characteristics and risk.
* **Behavioral Similarity to Default:** Indicators such as late payments, missed service charges, delinquent obligations, or recurring payment failures can resemble financial distress and therefore act as proxies for default risk.

Using a proxy enables the development of pre-screening and risk-ranking models that can support lending decisions even when traditional credit histories are unavailable.

### Business Risks of Proxy-Based Prediction

Although necessary, proxy variables introduce several business and operational risks:

* **Model Inaccuracy:** A proxy only approximates default behavior and may not fully capture actual credit risk.
* **Model Risk:** Incorrect assumptions about the relationship between the proxy and true default can lead to poor lending decisions.
* **Incomplete Financial Visibility:** Alternative data sources may observe revenue or transaction activity without capturing liabilities, expenses, or existing debt obligations.
* **Economic Fragility:** Proxy-based relationships established during stable economic periods may fail during recessions or financial crises.
* **Fraud and Manipulation Risks:** Some proxy indicators can be artificially influenced, leading to inaccurate risk assessments.
* **Operational Complexity:** Aggregating proxy information from multiple platforms may increase processing time and data management complexity.
* **Systemic Risk:** If many institutions rely on similar flawed proxies, errors can propagate across the financial system and contribute to broader instability.

Therefore, proxy-based models require careful validation, monitoring, and governance to ensure they remain reliable indicators of credit risk.

---

## 3. Trade-offs Between Logistic Regression with WoE and Gradient Boosting

Selecting a credit scoring model in a regulated financial environment requires balancing predictive performance with interpretability, transparency, and compliance requirements.

### Logistic Regression with Weight of Evidence (WoE)

#### Advantages

* Highly interpretable and easy to explain.
* Clear relationship between input variables and predicted risk.
* Regulatory-friendly due to transparency and auditability.
* Easier validation, monitoring, and documentation.
* Supports explainable lending decisions and adverse action reporting.
* Lower computational requirements.

#### Limitations

* Assumes largely linear relationships between predictors and outcomes.
* May not capture complex interactions or nonlinear patterns.
* Often produces lower predictive accuracy than advanced machine learning methods.

### Gradient Boosting Models (XGBoost, LightGBM, CatBoost)

#### Advantages

* Typically achieve superior predictive performance and higher AUC scores.
* Capture complex nonlinear relationships and feature interactions.
* Handle large datasets and diverse feature sets effectively.
* Often improve risk segmentation and borrower discrimination.

#### Limitations

* More difficult to interpret and explain.
* Can appear as "black-box" models to regulators and stakeholders.
* Require more extensive governance, validation, and monitoring.
* Higher computational and operational costs.
* Greater risk of overfitting if not properly controlled.

### Regulatory and Business Considerations

Financial regulations such as Basel II and model risk management frameworks emphasize transparency, explainability, fairness, and auditability. Consequently:

| Aspect                 | Logistic Regression + WoE | Gradient Boosting |
| ---------------------- | ------------------------- | ----------------- |
| Interpretability       | High                      | Low to Moderate   |
| Regulatory Acceptance  | High                      | More Challenging  |
| Documentation Effort   | Lower                     | Higher            |
| Predictive Performance | Moderate                  | High              |
| Computational Cost     | Low                       | Higher            |
| Auditability           | Strong                    | More Complex      |
| Overfitting Risk       | Lower                     | Higher            |

### Champion-Challenger Strategy

Many financial institutions address these trade-offs through a **champion-challenger approach**:

* The **champion model** is typically a traditional, highly interpretable model such as Logistic Regression with WoE.
* The **challenger model** is a more advanced machine learning model such as Gradient Boosting that is evaluated in parallel.

This strategy enables organizations to explore improvements in predictive performance while maintaining a transparent and regulatory-compliant baseline model. It provides a practical balance between innovation, governance, and risk management.
