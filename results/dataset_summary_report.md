# Disaster Tweet Classification - Dataset Summary & EDA Report

## 1. Split Sizes & Data Integrity
- **Total Tweets**: 76,484
- **Train Set**: 53,531 (70.0%)
- **Validation Set**: 7,793 (10.2%)
- **Test Set**: 15,160 (19.8%)
- **Missing Values**: 0 missing fields across all splits.
- **Exact Duplicates**: Train=0, Val=1, Test=1.
- **Cross-Split Overlap**: Train-Val Leakage=4, Train-Test Leakage=0.

## 2. Target Classes & Imbalance
- **Number of Classes**: 10
- **Imbalance Ratio**: **59.56:1** (Majority: `rescue_volunteering_or_donation_effort`, Minority: `missing_or_found_people`)
- **Strategy for F1 Optimization**: Apply inverse class frequency weighting $w_c = \frac{N}{|C| \cdot N_c}$ across all baseline and deep learning models to avoid starvation of low-frequency emergency classes (`missing_or_found_people`, `requests_or_urgent_needs`).

| Class Name | Train Count | Train % | Val Count | Test Count | Total Count | Total % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rescue_volunteering_or_donation_effort` | 14,891 | 27.82% | 2,168 | 4,219 | 21,278 | 27.82% |
| `other_relevant_information` | 8,501 | 15.88% | 1,236 | 2,407 | 12,144 | 15.88% |
| `sympathy_and_support` | 6,250 | 11.68% | 909 | 1,772 | 8,931 | 11.68% |
| `infrastructure_and_utility_damage` | 5,715 | 10.68% | 831 | 1,617 | 8,163 | 10.67% |
| `injured_or_dead_people` | 5,110 | 9.55% | 746 | 1,447 | 7,303 | 9.55% |
| `not_humanitarian` | 4,407 | 8.23% | 644 | 1,245 | 6,296 | 8.23% |
| `caution_and_advice` | 3,774 | 7.05% | 550 | 1,070 | 5,394 | 7.05% |
| `displaced_people_and_evacuations` | 2,800 | 5.23% | 409 | 790 | 3,999 | 5.23% |
| `requests_or_urgent_needs` | 1,833 | 3.42% | 264 | 521 | 2,618 | 3.42% |
| `missing_or_found_people` | 250 | 0.47% | 36 | 72 | 358 | 0.47% |

## 3. Text Length & Sequence Truncation Recommendations
- **Average Word Count**: 22.1 words
- **95th Percentile Word Count**: 44 words
- **99th Percentile Word Count**: 50 words
- **Max Sequence Length Recommendation**: Setting `max_length = 64` or `96` provides 100% token coverage for tweets while drastically accelerating Transformer and RNN inference.

## 4. Generated EDA Visualizations
- Class Distribution Plot: `results/eda_plots/class_distribution.png`
- Tweet Length KDE Plot: `results/eda_plots/text_length_distribution.png`
