<div align="center">
  <h1>Prediction avocado avg. price</h1>
 <p>(interview task)</p>
</div>

This excercise is part of interview process for Blue Berry. 

A model that was trained on data to predict avocado prices based on data.

## Analysis on the task
 First is this model (Linear regression) the best for this task?
 Yes... usually. There was a study that I found while I was preparing for which model to use for the task, using random forest (link: https://arxiv.org/abs/1605.00003). This study is refered in some cases to be used for market predictions, which was similar to my task, but unfortially due to constrains on the time given for the task this path was not persuit for the end result.

 To make the errors in prediction smaller, I have used the following steps:
 - Adding polynomial features (unfortunately max limit is 2 because of hardware limitations)
 - Decreasing regularization parameter
 - Increasing regularization parameter

Steps to limit the errors, not taken:
- get more training examples (currently there are ~18000 exampkes which I think is enough)
- try smaller set of features (due to time constrains this step was not pursuit)
- getting additional features (there are 70 features used for this data which is enough in my opinion)

The end result is currently the best operating model is with smaller regularization parameter and quadratic polynomial. Other models can be seen still in the code used as a comparison.

End remarks: the error is still high in my opinion and can be improved. Jtrain = 0.01769, Jcv = 0.02081, Jtest = 0.03842.


## Explanation on the code
From the date, it is used the day of the month and month.
```
# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract features from the date
data['DayOfMonth'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
```

With OneHotEncoder the data from categorical variables as numerical values.

In train_plot_poly first it is used just polynimial with linear.

In train_plot_reg_params it is used poly with reg_params.

And in the end  is used no poly and no reg_params.


 
