# titanic-survivors
<b>Titanic Survival Prediction Using Machine Learning</b>
In this project, we will analyze the Titanic data set and make predictions to see which passengers on board the ship would survive.
		The data is taken form kaagle.
After visualizing the data I noticed that preprocessing is required. Sex column as well as embraked column do not have the numerical values so I will first change them into numerical values by using labelencoding.
The next step is to split our dataset in to training data and testing data. We will do so by using train_test_split from sklearn.
The next step is to scale out data and we will scale our data by  using StandardScaler from sklearn.
Then we need to fit our training dataset into the machine learning model. We will fit into different models and analyse the prediction result to see which model works best for us. The machine learning models used here are LogisticRegression, KNeighborsClassifier, SupportVectorMachine, NaiveBayes, DecisionTree and RandomForest. 
Last step is to predict whether a passenger survives or not. After predicting we use confusion matrix to determine the parameters such as Accuracy, Recall, Precision, F-measure etc.