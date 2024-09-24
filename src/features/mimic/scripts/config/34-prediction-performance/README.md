## #34 Data extraction for prediction performance

### Naming conventions

Extractions will be named using the following convention: 

`[condition_abbreviation]_B[backwards_window_value][backwards_window_unit]_G[value][unit]_P[value][unit]`

For example:

`t2d_B2y_G3m_P3m`: Type 2 diabetes, backwards window of 2 years, gap of 3 months, prediction window of 3 months.

The following tables list the abbreviations:

| Condition                             | Abbreviation |
|---------------------------------------|--------------|
| Type 2 Diabetes                       | `t2d`        |
| Chronic Kidney Disease Stage 4 or 5   | `ckd45`      |
| Acute Myocardial Infarction         ` | `ami`        |

| Time unit | Symbol |
|-----------|--------|
| d         | day    |
| w         | week   |
| m         | month  |
| y         | year   |
