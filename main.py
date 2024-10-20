import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv('data\\avocado.csv')

# Convert Date to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Extract features from the date
data['DayOfMonth'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month

#Extract the columns with object datatype as they are the categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

#Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply one-hot encoding to the categorical columns
one_hot_encoded = encoder.fit_transform(data[categorical_columns])

#Create a DataFrame with the one-hot encoded columns
#We use get_feature_names_out() to get the column names for the encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the one-hot encoded dataframe with the original dataframe
df_encoded = pd.concat([data, one_hot_df], axis=1)

# Drop the original categorical columns
df_encoded = df_encoded.drop(categorical_columns, axis=1)


# Select features for prediction
features = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year', 'Month', 'DayOfMonth', 'region']
X = df_encoded.iloc[:, 3:].values
y = df_encoded.iloc[:, 2].values

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=80)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=80)

del X_, y_

# Instantiate the regression model class
model = LinearRegression()

def train_plot_poly(model, x_train, y_train, x_cv, y_cv, x_test, y_test, max_degree=2):
    train_mses = []
    cv_mses = []
    models = []
    scalers = []
    degrees = range(1, max_degree + 1)

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for degree in degrees:
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

        # Compute the test
        X_test_mapped = poly.fit_transform(x_test)
        X_test_mapped_scaled = scaler_poly.transform(X_test_mapped)
        yhat_test = model.predict(X_test_mapped_scaled)
        mse = mean_squared_error(y_test, yhat_test)

        print(y_test)
        print(f"Predicted Average Price with poly: ${yhat_test}")
        print(mse)

    # Plot the results
    plt.plot(degrees, train_mses, marker='o', c='r', label='training MSEs');
    plt.plot(degrees, cv_mses, marker='o', c='b', label='CV MSEs');
    plt.title("degree of polynomial vs. train and CV MSEs")
    plt.xticks(degrees)
    plt.xlabel("degree");
    plt.ylabel("MSE");
    plt.legend()
    plt.show()

def train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, x_test, y_test, degree=1):
    train_mses = []
    cv_mses = []
    models = []
    scalers = []

    # Loop over 10 times. Each adding one more degree of polynomial higher than the last.
    for reg_param in reg_params:
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Scale the training set
        scaler_poly = StandardScaler()
        X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
        scalers.append(scaler_poly)

        # Create and train the model
        model = Ridge(alpha=reg_param)
        model.fit(X_train_mapped_scaled, y_train)
        models.append(model)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped_scaled)
        train_mse = mean_squared_error(y_train, yhat) / 2
        train_mses.append(train_mse)

        # Add polynomial features and scale the cross-validation set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_cv_mapped = poly.fit_transform(x_cv)
        X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped_scaled)
        cv_mse = mean_squared_error(y_cv, yhat) / 2
        cv_mses.append(cv_mse)

        # Compute the test
        X_test_mapped = poly.fit_transform(x_test)
        X_test_mapped_scaled = scaler_poly.transform(X_test_mapped)
        yhat_test = model.predict(X_test_mapped_scaled)

        mse = mean_squared_error(y_test, yhat_test)

        print(y_test)
        print(f"Predicted Average Price with reg: ${yhat_test}")
        print(mse)

    # Plot the results
    reg_params = [str(x) for x in reg_params]
    plt.plot(reg_params, train_mses, marker='o', c='r', label='training MSEs');
    plt.plot(reg_params, cv_mses, marker='o', c='b', label='CV MSEs');
    plt.title("lambda vs. train and CV MSEs")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

reg_params = [0.00000001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

# Define degree of polynomial and train for each value of lambda
train_plot_poly(model, X_train, y_train, X_cv, y_cv, X_test, y_test, max_degree=2)
train_plot_reg_params(reg_params, X_train, y_train, X_cv, y_cv, X_test, y_test, degree=2)

model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Example usage
print(y_test)
print(f"Predicted Average Price simple: ${y_pred}")

