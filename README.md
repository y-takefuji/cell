# cell
<pre>
$ python cv3-109f.py
| Method | Top 10 accuracy±std | Top 9 accuracy±std | Consistency |           
| HVGS | 0.8794 ± 0.0215 | 0.8809 ± 0.0187 | Pass |
| FA | 0.8780 ± 0.0164 | 0.8823 ± 0.0159 | Pass |
| PCA | 0.8681 ± 0.0035 | 0.8638 ± 0.0053 | Fail |

$python cv4-5f.py
PCA - 5-fold CV Accuracy: 0.7730 ± 0.0311
ICA - 5-fold CV Accuracy: 0.8596 ± 0.0053
HVGS - 5-fold CV Accuracy: 0.8794 ± 0.0215
Feature Agglomeration - 5-fold CV Accuracy: 0.8667 ± 0.0069

$python cv5f10.py
Lasso - 5-fold CV Accuracy: 0.8752 ± 0.0146
Logistic - 5-fold CV Accuracy: 0.8794 ± 0.0135
PCA - 5-fold CV Accuracy: 0.8695 ± 0.0085
HVGs - 5-fold CV Accuracy: 0.8794 ± 0.0215
Random Forest - 5-fold CV Accuracy: 0.8851 ± 0.0113

$python cv5f9.py
Lasso - 5-fold CV Accuracy: 0.8809 ± 0.0122
Logistic - 5-fold CV Accuracy: 0.8780 ± 0.0122
PCA - 5-fold CV Accuracy: 0.8809 ± 0.0138
HVGs - 5-fold CV Accuracy: 0.8809 ± 0.0187
Random Forest - 5-fold CV Accuracy: 0.8922 ± 0.0138



</pre>
