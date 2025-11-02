# %% Generate 2nd order polynomial features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from matplotlib import pyplot as plt

start = -2
end = 2
X = np.linspace(start, end, 1000)
X2 = np.linspace(start,end, 100)
def y_true(X):
    return X**2 + 2*X + np.random.normal(0,1, len(X))
Y = y_true(X)
y_min = np.min(Y)
y_max = np.max(Y)


# randomly select 30 points from the X
degrees = [1, 2, 10]
num_models_per_degree = 20
N_train = 200
N_test = 20
# create subplot with 1 row and 5 columns
fig, axs = plt.subplots(1, len(degrees), figsize=(20, 4))
# for each degree calculate the mse for 30 models
mse = {}
# select a test set of 20 points
X_test = np.random.choice(X, N_test, replace=False)
y_test = y_true(X_test)
# set X_train to be the rest of the points
X_train = np.setdiff1d(X, X_test)

for i, degree in enumerate(degrees):
    mse[degree] = []
    for j in range(num_models_per_degree):
        # select N points from X_train
        X_train_sample = np.random.choice(X_train, N_train, replace=False)
        X_train_sample = np.sort(X_train_sample)

        y_true_sample = y_true(X_train_sample)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train_sample[:, np.newaxis], y_true_sample)
        y_pred = model.predict(X_train_sample[:, np.newaxis])
        


        # calculate mse on the test set
        y_test_pred = model.predict(X_test[:, np.newaxis])
        mse[degree].append(np.mean((y_test-y_test_pred)**2))
        if j == 0:
            line_model = axs[i].plot(X2, model.predict(X2[:, np.newaxis]), 'r-', alpha=0.2)
        else:
            axs[i].plot(X2, model.predict(X2[:, np.newaxis]), 'r-', alpha=0.2)
        axs[i].plot(X_train_sample, y_true_sample, 'o', alpha=0.2)
    # add text with mse average
    # plot the test set
    line_test = axs[i].plot(X_test, y_test, 's', markersize=10)
    line_train = axs[i].plot(X_train_sample, y_true_sample, 'o', alpha=0.2)
    axs[i].set_title(f'Degree {degree}, Training Set Size: {N_train}')
    # add a legend that shows the model, the test set and the training set using the same colors and markers
    mse_mean_ax = np.mean(mse[degree])
    mse_std_ax = np.std(mse[degree])
    mse_mean_text = f'MSE: {mse_mean_ax:.2f}' if mse_mean_ax < 10000 else f'MSE: >10,000'
    mse_var_text = f'Var: {mse_std_ax:.2f}' if mse_std_ax < 1000 else f'Var: >10,000'
    axs[i].legend([line_model[0], line_test[0], line_train[0]], [f'Model, {mse_mean_text}, {mse_var_text}', 'Test Set', 'Training Set'], fontsize=14)
    # set axis limits
    axs[i].set_ylim(y_min, y_max)
   
# %%
