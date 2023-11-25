# Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning
Predict Clicked Ads Customer using Machine Learning

# Background and Objective
A company wants to know the effectiveness of the advertisements they broadcast. This is important for the company to know how successful the advertisement being marketed is so that it can attract customers to see the advertisement. By processing historical advertisement data and finding insights and patterns that occur, can help companies determine marketing targets. The focus of this case is to create a machine learning classification model that functions to determine the right target customers.

# Scope of problem



# Data

# Data Preprocessing
1. Handling missing values ​​will be carried out on features:
- Daily Time Spent on Site, filled in with the median
- Area Income, fill in the median
- Daily Internet Usage, fill in the median
- Gender, filled in with mode

2. Create a new feature from timestamp to year, month, week and day and change week data type into int64
3. Perform One-Hot Encoding for the "Gender" feature and replace "1" with "Yes" and "0" with "No" in the "Clicked on Ad" feature.
4. Delete the following features: "Unnamed Features: 0" since it doesn't provide any information, "year" as it contains only one unique value, "Timestamp" because feature engineering has already been performed, and "city," "province," and "category" due to the high number of unique values which would lead to the curse of dimensionality when one-hot encoding is applied.
5. Split with 70% Data Training and 30% Data Testing
6. Outliers will be removed in the Income Area in Data Train using Z-Scores (Before = 700 data, After = 699 data)
7. Data Training and Data Test has Class Balanced

# Data Analysis
## Statistical Analysis
1. The contents of all columns are reasonable
2. The maximum/minimum value is not too far from the mean/median
3. The mean and median are not too far apart
4. The contents of each unique value are reasonable

## Univariate Analysis - Numerical
1. There are outliers in the income feature area
2. Feature Area Income distribution is slightly Negative Skew
3. The Daily Time Spent distribution feature includes bimodal distribution
4. Feature Age distribution is slightly positive Skew
5. The daily internet usage feature has a normal distribution
6. Characteristics of customers who click on ads
7. Characteristics of customers who click on ads
- Daily Time Spent on site = 40 – 60
- Age = 35 – 50
- Area Income = 3 – 4
- Daily Internet Usage = 120 - 150
8. Characteristics of customers who don't click on ads
- Daily Time Spent on site = 75 - 85
- Age = 25 - 35
- Area Income = 4 - 5
- Daily Internet Usage = 220 – 240
9. Customers who click on ads tend to spend more daily time on site than those who don't click on ads
10. Customers who click on ads tend to be older than those who don't click on ads
11. Customers who click on ads tend to have a lower income area than those who don't click on ads
12. Customers who click on ads tend to have lower daily internet usage than those who don't click on ads

## Univariate Analysis - Categorical
1. There are more women than men, although not much different
2. The largest province is DKI Jakarta
3. Automotive is the highest category, but not much different from other categories
4. The largest cities are Surabaya and Bandung
5. If grouped by gender, women click more on Ads than men
6. If grouped by city, some cities such as Cimahi and Serang tend to click on ads compared to other cities
7. If grouped by province, South Kalimantan and Banten tend to click on ads compared to other provinces
8. If grouped by category, Finance and Fashion tend to click on ads compared to other categories

## Bivariate Analysis
1. Customers who click on ads tend to be more mature with daily time spent on the site, daily internet usage, and low area income
2. It can be concluded that the longer the customer, the lower the daily time spent on the site, daily internet usage and area income, and vice versa
3. There are 4 features that have a high correlation with the target (Clicked on Ad)
4. Age is negatively correlated with Daily Time Spent, Area Income and Daily Internet Usage. However, it has a positive correlation with the target (Clicked on Ad)
5. Are Income is positively correlated with Daily Time Spent and Daily Internet Usage. However, it has a negative correlation with the target (Clicked on Ad)
6. Daily Internet Usage is positively correlated with Daily Time on Site. However, it has a negative correlation with the target (Clicked on Ad)
7. Daily Time on Site has a negative correlation with the target (Clicked on Ad)

# Metrics Evaluation
The chosen evaluation metric is Recall, which the goals to reduce False Negative (Predict No actually they click the Ads), because the main is to maximizing the number of customers who genuinely click on the ad to enhance the effectiveness of advertising delivery.

# Data Modeling

## Experiment 1: Modeling without Normalization/Standardization

![Picture1](https://github.com/pwirap/Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning/assets/99533745/bcb7f347-1ab3-43c5-9fac-48b9e8500154)

From the comprehensive analysis of model evaluations, it's apparent that a substantial gap exists between the training and test data, signaling potential overfitting. Initial experiments highlight Gradient Boosting and Random Forest's robust recall performance, while SVM shows the lowest recall compared to the other models.


## Experiment 2: Modeling with Normalization/Standardization

![Picture2](https://github.com/pwirap/Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning/assets/99533745/5fda8abe-6738-4e8d-a818-a75f5f9a33d9)

After standardization, certain features like 'Daily Time Spent on Site,' 'Daily Internet Usage,' 'Area Income,' and 'Age' notably improved evaluation scores for SVM and KNN, while showing minimal impact on Gradient Boosting, Random Forest, Decision Tree, and XGBoost. Subsequent experiments revealed consistent strong recall performance in Gradient Boosting and Random Forest, leading to the selection of Random Forest as the top-performing model. However, the discernible gap between training and test data suggests a need for hyperparameter tuning to mitigate overfitting.

# Selected Model

![Picture3](https://github.com/pwirap/Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning/assets/99533745/4f99370e-eff8-40b0-9a22-57ef187ab8a7)

Among the various machine learning models that have been explored, Random Forest has been selected as the top-performing model. Following this, hyperparameter tuning will be conducted after standardization to mitigate the risk of overfitting. The table besides show the Evaluation Matrix after doing hyperparameter tuning.

# Hyperparameter Tuning

Best n_estimators: 1600
Best bootstrap: False
Best criterion: gini
Best max_depth: 6
Best min_samples_split: 82
Best min_samples_leaf: 55
Best max_features: auto
Best n_jobs: -1

Using RandomizedSearchCV, table besides show the best hyperparameter tuning

# Confusion Matrix for Best Model (Random Forest)

![Picture4](https://github.com/pwirap/Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning/assets/99533745/c55517d6-4867-4862-a2e2-f344ba375aa7)

As mentioned earlier with a focus on recall, from the confusion matrix, it is apparent that the number of False Negatives (Predictive no actual they click the ads)  has been significantly reduced, aligning with the previously provided Evaluation Matrix table where the Recall was at 96%.

# Feature Importance

![Picture5](https://github.com/pwirap/Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning/assets/99533745/822df290-0a9b-46e8-857d-732ac7f05b69)

After determining the best model and conducting hyperparameter tuning, a feature importance analysis was performed. As depicted in the Feature Important Score graph, it is evident that Daily Internet Usage and Daily Time Spent on Site emerge as the top two most crucial features for optimizing marketing success

# Business Recommendation

1. Enhance Ad Relevance for Customer Loyalty: Customers who click on ads tend to spend more daily time on the site, indicating their interest. Tailor ads with more relevant and engaging content to boost customer loyalty.
2. Age-Driven Segmentation: If older customers are more likely to click on ads, adapt ads to attract this demographic. Offer products and promotions tailored to their preferences and needs.
3. Income-Based Ad Offers: Focus on optimizing ads for lower-income areas where customers tend to click more. Offer affordable products and services to align with their spending capacity.
4. Target More Mature Audience: Develop advanced ad strategies for mature customers who spend more time on the site, have lower daily internet usage, and reside in low-income areas. Consider premium or exclusive offers for this segment.
5. Adapt Ads to Popular Categories: Optimize ads in categories like Finance and Fashion, which receive more clicks. Offer specialized products or promotions that resonate with customer interests in these categories.
6. Engage Low Daily Internet Users: Create captivating ads for customers with lower daily internet usage. Focus on offline or simpler device-centric offers to cater to their preferences.

# Business Simulation

![image](https://github.com/pwirap/Predict_Clicked_Ads_Customer_Classification_by_Using_Machine_Learning/assets/99533745/2f47e269-2450-44fd-b542-ef83b12b0df6)

Machine Learning can work well to increase revenue which in turn increases Ads Marketing efficiency 
