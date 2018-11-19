The python file contains the system that attempts to solve a binary classification problem based on the dataset provided by Quartic.ai.
The predict.csv file contains the predicted values from the system.

DATASET-
The training data provided contains 596000 entries each having a feature vector of 55 entries. The features are divided into 3 types-
Numerical features- 23,
Derived features- 19,
Categorical features- 14.
The training dataset is in itself an unbalanced dataset with 574284 entries of class 0 and rest 21716 entries of class 1 creating an almost 1:26 bias.
The test data has 892816 entries having 55 features each.

SYSTEM APPROACH-
Taking the unbalanced nature of the dataset into account the most appropriate approch would be use a form of outlier detector for classification. One-class SVM and Isolation Forest are the preferred method for such a problem. But on actually testing this on the training data it was seen that Isolation Forest(IF) was overfitting on the majority class. As for One-class SVM (OCSVM) the huge dataset made it impossible for me to process the entire dataset in one go on my i5 system. A bagging ensemble of OCSVMs was still overfitting on the majority class. All this indicated some sort of structure in the minority class telling that the minority class are not outliers but a second separate class with it's own properties.
In order to balance the dataset, SMOTE is used to synthetically generate dataitems for the minority class. This added nearly 550000 data entries of class 1 making the new modified dataset of 1148568 entries. Again, this huge dataset was impossible to fit in a single classifier but a restricted dataset of 75000 entries on a rbf SVM kernel provided around 67% accuracy on a validation set derived from the modified dataset. In order to actually compute the predictions here an ensemble of 20 linear SVM kernels are used each training on 1/20th the data. This results in a high bias towards class 1.

IMPROVEMENTS-
The accuracy of system can be improved by using a single classifer in place of a bagging ensemble, further more a cross reference with the original dataset(without SMOTE entries) on OCSVM can be used to reliably predict the minority class. A correlation in the features can also be studied and some dimensionality reduction can be used. All this would require a better processor (possible a dedicated server) and it is quite impratical to test all these on my i5 system. Since the dataset is huge deep learning would also provide impressive results and with appropriate curation a NEAT genetic algorithm system might we able to train well on this data.
