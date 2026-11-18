# PathSnap — Comparative Model Analysis

## 1. Overall model comparison

All five models were evaluated on the same held-out set of **186,368 candidate rows** from **10 trips**.

| Rank | Model | Accuracy | Precision | Recall | Macro F1 | Top-1 Accuracy | Log Loss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **HistGradientBoosting** | **93.14%** | **66.99%** | **62.51%** | **80.44%** | **66.17%** | **0.1786** |
| 2 | Random Forest | 92.30% | 62.79% | 57.43% | 77.87% | 59.92% | 0.2854 |
| 3 | KNN | 91.08% | 55.33% | 58.38% | 75.92% | 56.54% | 0.7105 |
| 4 | Decision Tree | 88.14% | 42.84% | 53.88% | 70.52% | 53.67% | 4.2755 |
| 5 | Logistic Regression | 70.08% | 24.16% | 92.39% | 59.28% | 61.68% | 0.5418 |

## 2. Model-wise analysis

| Model | What the results show | Strength | Limitation | Assessment |
|---|---|---|---|---|
| **HistGradientBoosting** | Highest accuracy (93.14%) and macro F1 (80.44%); lowest log loss (0.1786). | Best overall balance in this comparison. | Recall of 62.51% means some positive candidates are missed. | **Selected final model** |
| Random Forest | Second-highest accuracy (92.30%) and macro F1 (77.87%). | Strong alternative. | Lower recall and macro F1 than HistGradientBoosting. | Runner-up |
| KNN | Accuracy 91.08%; macro F1 75.92%. | Competitive result. | Lower precision and top-1 accuracy; higher log loss. | Third |
| Decision Tree | Accuracy 88.14%; macro F1 70.52%. | Relatively easy to explain. | Lower precision and highest log loss (4.2755). | Not selected |
| Logistic Regression | Highest recall (92.39%). | Finds a large share of positive labels. | Low precision (24.16%) and lowest macro F1 (59.28%). | Not selected |

## 3. Metric-wise analysis

| Metric | Best model | Result | Interpretation |
|---|---|---:|---|
| Accuracy | HistGradientBoosting | **93.14%** | Highest overall proportion of correctly classified candidate rows. |
| Precision | HistGradientBoosting | **66.99%** | Highest proportion of predicted positives that match positive supplied labels. |
| Recall | Logistic Regression | **92.39%** | Identified the largest share of positive labels, but had low precision. |
| Macro F1 | HistGradientBoosting | **80.44%** | Best balance of precision and recall across both classes. |
| Top-1 accuracy | HistGradientBoosting | **66.17%** | Highest rate of ranking the correct candidate first within a GPS-point group. |
| Log loss | HistGradientBoosting | **0.1786** | Lowest probability-prediction loss; lower is better. |

## 4. Training and evaluation setup

| Item | Details |
|---|---|
| Dataset | `all_train_candidates.csv` |
| Total candidate rows | 1,667,336 |
| Number of trips | 80 |
| Training split | 70 trips |
| Held-out split | 10 trips |
| Held-out candidate rows | 186,368 |
| Input features | `distance_to_road`, `heading_difference`, `speed`, `time_difference` |
| Target | `label` |
| Grouping key | `trip_id` |
| Split method | Trip-grouped holdout (`GroupShuffleSplit`) |
| Random state | `42` |

### Training sample sizes used for the comparison

| Model | Rows used to fit comparison model |
|---|---:|
| Logistic Regression | 250,000 |
| Decision Tree | 250,000 |
| Random Forest | 250,000 |
| HistGradientBoosting | 250,000 |
| KNN | 50,000 |

All five models were scored on the same held-out rows. Different fitting sample sizes were used to keep the comparison computationally practical.

## 5. Final model decision

| Decision factor | Result |
|---|---|
| Selected model | **HistGradientBoosting** |
| Reason | Highest macro F1, accuracy, precision, and top-1 accuracy; lowest log loss. |
| Macro F1 | **80.44%** |
| Accuracy | **93.14%** |
| Final training step | The saved HistGradientBoosting pipeline was refitted using all available rows from the 70 training trips. |
| Metrics caveat | The comparison metrics above are from the 250,000-row comparison fit, not a separate evaluation of the final refitted pipeline. |

## 6. Dataset notes and limitations

| Item | Observation |
|---|---|
| Class distribution | 88.73% negative labels; 11.27% positive labels. |
| Missing values | 0 in the inspected dataset. |
| Validation design | One trip-grouped holdout comparison; five-fold cross-validation was not completed. |
| Label meaning | Labels are derived from the supplied reference-route data. |
| Metric interpretation | Scores measure agreement with supplied labels, not independently verified real-world map-matching accuracy. |

## 7. Review summary

| Question | Answer |
|---|---|
| What was compared? | Logistic Regression, KNN, Decision Tree, Random Forest, and HistGradientBoosting. |
| Which features were used? | Distance to road, heading difference, speed, and time difference. |
| How was testing performed? | 70 trips for training and 10 separate trips held out for evaluation. |
| Which model was selected? | **HistGradientBoosting**. |
| Why was it selected? | It achieved the highest macro F1-score of **80.44%** on the held-out comparison. |

---

*PathSnap project review — Person 2: model selection and training.*
