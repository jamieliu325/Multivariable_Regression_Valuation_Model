import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np



# set up data display format
pd.options.display.float_format = '{:,.2f}'.format

# read csv file via pandas
data = pd.read_csv('boston.csv', index_col=0)

def data_exploration():
    print(data.shape)
    print(data.columns)
    print(data.head())
    print(data.tail())
    print(data.count())

def data_cleaning():
    print(data.info())
    print(f'Any NaN values? {data.isna().values.any()}')
    print(f'Any duplicates? {data.duplicated().values.any()}')

def descriptive_statistics():

    print(data.describe())

    sns.displot(data.PRICE, bins=50, aspect=2, kde=True, color='#2196f3')
    plt.title(f'1970s Home Values in Boston. Average: ${(1000*data.PRICE.mean()):.6}')
    plt.xlabel('Price in 000s')
    plt.ylabel('Nr. of Homes')
    plt.show()

    sns.displot(data.DIS,bins=50,aspect=2,kde=True,color='darkblue')
    plt.title(f'Distance to Employment Centres. Average: {(data.DIS.mean()):.2}')
    plt.xlabel('Weighted Distance to 5 Boston Employment Centres')
    plt.ylabel('Nr. of Homes')
    plt.show()

    sns.displot(data.RM,aspect=2,kde=True,color='#00796b')
    plt.title(f'Distribution of Rooms in Boston. Average; {data.RM.mean():.2}')
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Nr. of Homes')
    plt.show()

    plt.figure(figsize=(10,5),dpi=200)
    plt.hist(data.RAD,bins=24,ec='black',color='#7b1fa2',rwidth=0.5)
    plt.xlabel('Accessibility to Highways')
    plt.ylabel('Nr. or Houses')
    plt.show()

    river_access = data.CHAS.value_counts()
    bar = px.bar(x=['No', 'Yes'], y=river_access.values,color=river_access.values,color_continuous_scale=px.colors.sequential.haline,title='Next to Charles River?')
    bar.update_layout(xaxis_title='Property Located Next to the River?',yaxis_title='Number of Homes',coloraxis_showscale=False)
    bar.show()

def data_relatioships():

    sns.pairplot(data)
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data.DIS,y=data.NOX,height=8,kind='scatter',color='deeppink',joint_kws={'alpha':0.5})
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data.NOX,y=data.INDUS,height=7,color='darkgreen',joint_kws={'alpha':0.5})
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data.LSTAT,y=data.RM,height=7,color='orange',joint_kws={'alpha':0.5})
    plt.show()

    with sns.axes_style('darkgrid'):
        sns.jointplot(x=data.LSTAT,y=data.PRICE,height=7,color='crimson',joint_kws={'alpha':0.5})
    plt.show()

    with sns.axes_style('whitegrid'):
        sns.jointplot(x=data.RM,y=data.PRICE,height=7,color='darkblue',joint_kws={'alpha':0.5})
    plt.show()

def multivariable_regression():

    # split training and test dataset
    target=data.PRICE
    features=data.drop('PRICE',axis=1)
    X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=10)
    train_pct = 100*len(X_train)/len(features)
    print(f'Training data is {train_pct:.3}% of the total data')
    test_pct=100*X_test.shape[0]/features.shape[0]
    print(f'Test data makes up the remaining {test_pct:.3}%.')

    # multivariable regression
    regr = LinearRegression()
    regr.fit(X_train,y_train)
    rsquared = regr.score(X_train,y_train)
    print(f'Training data r-squared: {rsquared:.2}')

    # evaluate the coefficients of the model
    regr_coef = pd.DataFrame(data=regr.coef_,index=X_train.columns,columns=['Coefficient'])
    print(regr_coef)

    # Premium for having an extra room
    premium = regr_coef.loc['RM'].values[0]*1000
    print(f'The price premium for having an extra room is ${premium:.5}')

    # analyse the estimated values & regression residuals
    predicted_vals = regr.predict(X_train)
    residuals = (y_train - predicted_vals)

    plt.figure(dpi=100)
    plt.scatter(x=y_train,y=predicted_vals,c='indigo',alpha=.6)
    plt.plot(y_train,y_train,color='cyan')
    plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
    plt.xlabel('Predicted Prices $\hat y_i$', fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.show()

    plt.figure(dpi=100)
    plt.scatter(x=predicted_vals,y=residuals,c='indigo',alpha=0.6)
    plt.title('Residuals vs predicted Values,', fontsize=17)
    plt.xlabel('Predicted Prices $\hat y _i$',fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.show()

    resid_mean = round(residuals.mean(),2)
    resid_skew = round(residuals.skew(),2)
    sns.displot(residuals,kde=True,color='indigo')
    plt.title(f'Redisuals Skew ({resid_skew}) Mean ({resid_mean})')
    plt.show()

    # data transformations for a better fit
    tgt_skew = data.PRICE.skew()
    sns.displot(data.PRICE,kde=True,color='green')
    plt.title(f'Normal Prices. Skew is {tgt_skew:.3}')
    plt.show()

    y_log = np.log(data.PRICE)
    sns.displot(y_log,kde=True)
    plt.title(f'Log Prices. Skew is {y_log.skew():.3}')
    plt.show()

    plt.figure(dpi=150)
    plt.scatter(data.PRICE,np.log(data.PRICE))
    plt.title('Mapping the Original Price to a Log Price')
    plt.ylabel('Log Price')
    plt.xlabel('Actual $ Price in 000s')
    plt.show()

    # regression using log prices
    new_target = np.log(data.PRICE)
    features = data.drop('PRICE',axis=1)
    X_train,X_test,log_y_train,log_y_test = train_test_split(features,new_target,test_size=0.2,random_state=10)
    log_regr=LinearRegression()
    log_regr.fit(X_train,log_y_train)
    log_rsquared=log_regr.score(X_train,log_y_train)
    log_prediction=log_regr.predict(X_train)
    log_residuals=(log_y_train-log_prediction)
    print(f'Training data r-squared: {log_rsquared:.2}')

    # evaluate coefficients with log prices
    df_coef = pd.DataFrame(data=log_regr.coef_,index=X_train.columns,columns=['coef'])
    print(df_coef)

    # regression with log prices & residual plots
    plt.scatter(x=log_y_train,y=log_prediction,c='navy',alpha=0.6)
    plt.plot(log_y_train,log_y_train,color='cyan')
    plt.title(f'Actual vs Predicted Log Prices: $y_i$ vs $\hat y_i$ (R-Squared {log_rsquared:.2})',fontsize=17)
    plt.xlabel('Actual Log Prices $y_i$',fontsize=14)
    plt.ylabel('Predicted prices 000s $\hat y_i$', fontsize=14)
    plt.show()

    plt.scatter(x=y_train,y=predicted_vals,c='navy',alpha=0.6)
    plt.plot(y_train,y_train,color='cyan')
    plt.title('Original Actual vs Predicted Prices: $y_i$ vs $\hat y_i$ (R-Squared {log_rsquared:.2})', fontsize=17)
    plt.xlabel('Actual Prices 000s $y_i$', fontsize=14)
    plt.ylabel('Predicted Prices 000s $\hat y_i$', fontsize=14)
    plt.show()

    plt.scatter(x=log_prediction, y=log_residuals, c='navy', alpha=0.6)
    plt.title('Residuals vs Fitted Values for Log Prices', fontsize=17)
    plt.xlabel('Predicted Log Prices $\hat y_i$', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.show()

    plt.scatter(x=predicted_vals,y=residuals,c='indigo',alpha=0.6)
    plt.title('Original Redisual vs FItted Values', fontsize=17)
    plt.xlabel('Predicted Prices $\hat y_i$',fontsize=14)
    plt.ylabel('Residuals',fontsize=14)
    plt.show()

    log_resid_mean = round(log_residuals.mean(),2)
    log_resid_skew = round(log_residuals.skew(),2)

    sns.displot(log_residuals,kde=True,color='navy')
    plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')
    plt.show()

    sns.displot(residuals,kde=True,color='indigo')
    plt.title(f'Original model: Residuals Skew ({resid_skew}) Mean ({resid_mean})')
    plt.show()

    print(f'Original Modle Test Data r-squaured: {regr.score(X_test,y_test):.2}')
    print(f'Log Model Test Data r-squared {log_regr.score(X_test,log_y_test):.2}')

    # predict property value using the regression coefficients
    features = data.drop('PRICE',axis=1)
    average_vals = features.mean().values
    property_stats = pd.DataFrame(data=average_vals.reshape(1,len(features.columns)),columns=features.columns)
    print(property_stats)

    log_estimate = log_regr.predict(property_stats)[0]
    print(f'The og price estimate is ${log_estimate:.3}')
    dollar_est = np.exp(log_estimate)*1000
    print(f'The property is estimated to be worth ${dollar_est:.6}')

    # define property characteristics
    next_to_river = True
    nr_rooms = 8
    students_per_classroom = 20
    distance_to_town = 5
    pollution = data.NOX.quantile(q=0.75)
    amount_of_poverty=data.LSTAT.quantile(q=0.25)

    property_stats.RM = nr_rooms
    property_stats.PTRATIO = students_per_classroom
    property_stats.DIS = distance_to_town
    if next_to_river:
        property_stats.CHAS=1
    else:
        property_stats.CHAS=0
    property_stats.NOX=pollution
    property_stats.LSTAT=amount_of_poverty

    log_estimate=log_regr.predict(property_stats)[0]
    print(f'The log price estimate is ${log_estimate:.3}')
    dollar_est = np.exp(log_estimate)*1000
    print(f'The property is estiamed to be worth ${dollar_est:.6}')
