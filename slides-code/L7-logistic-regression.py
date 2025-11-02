# %%
import numpy as np
import matplotlib.pyplot as plt

# %% 
# create a plot with a line y = x
xlim = 5
x = np.linspace(0, xlim, 100)
y = x
# make the line width 2
plt.plot(x, y, linewidth=2)

# add a dashed line y = 1, y=2 and y=3
plt.plot(x, np.ones_like(x), '--')
plt.plot(x, 2*np.ones_like(x), '--')
plt.plot(x, 3*np.ones_like(x), '--')

# add a dashed vertical line at x=3
plt.axvline(x=3, color='grey', linestyle='--')

# set the x label to "cs_stats_score"
plt.xlabel('cs_stats_score')

# set the y label to "predicted major"
plt.ylabel('Predicted Major')

# set x limits to 0 and 100
plt.xlim([0, xlim])
# %%
# create a sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# create create a heat map of two variables with the color indicating the value of the sigmoid function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = sigmoid(X + Y)
# create a heat map
plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='coolwarm')
plt.colorbar()
# %%
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = sigmoid(X**2 + Y)
# create a heat map
plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='coolwarm')
# make the color bar range from 0 to 1
plt.clim(0, 1)
plt.colorbar()
# %%
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
# values from logistic regression below
c = [-0.00696363,  0.40005411,  0.30417498, -3.26357922, -0.16415006, -3.69250907]
Z = sigmoid(3.8+c[0] + c[1]*X + c[2]*Y + c[3]*X**2 + c[4]*X*Y + c[5]*Y**2)
# c = np.ones(6)
# Z = sigmoid(-5+c[0] + c[1]*X + c[2]*Y + c[3]*X**2 + c[4]*X*Y + c[5]*Y**2)
# create a heat map
# set x and y labels
plt.xlabel('X1')
plt.ylabel('X2')
plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='coolwarm')
# make the color bar range from 0 to 1
plt.clim(0, 1)
plt.colorbar()

# %%
# plot the sigmoid function
x = np.linspace(-5, 5, 100)
y = sigmoid(x)
plt.plot(x, y, linewidth=2)
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
# add scatter data with y = 1 if x > 0, y = 0 if x < 0
# generate random x between -5 and 5
x = np.random.uniform(-5, 5, 100)
y = (x > 0).astype(int)
# if y = 0, plot the point in red, otherwise plot the point in blue
plt.scatter(x[y==0], y[y==0], color='red')  
plt.scatter(x[y==1], y[y==1], color='blue')
# add a vertical line at x = 0
plt.axvline(x=0, color='grey', linestyle='--')

# %% plot decision boundary for 2D data using logistic regression, include the data points
from sklearn.linear_model import LogisticRegression

# generate random data
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
y_train = y
x_train = X
# fit logistic regression
lr = LogisticRegression()
lr.fit(X, y)

# plot the decision boundary
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = lr.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
plt.contourf(X, Y, Z, alpha=0.3)
# plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
# plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.scatter(x_train[y_train==0, 0], x_train[y_train==0, 1], color='red')
plt.scatter(x_train[y_train==1, 0], x_train[y_train==1, 1], color='blue')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')


# %%
# create data within a circle of radius 1
np.random.seed(0)
X = np.random.randn(200, 2)
y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)

# plot the data
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.scatter(X[y==0, 0], X[y==0, 1], color='red')

# fit logistic regression
lr = LogisticRegression()
lr.fit(X, y)

# plot the decision boundary
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = lr.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
plt.contourf(X, Y, Z, alpha=0.3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')

# %%
# fit logistic regression with polynomial features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
np.random.seed(0)
X = np.random.randn(200, 2)
y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)

# plot the data
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.scatter(X[y==0, 0], X[y==0, 1], color='red')

# fit logistic regression with polynomial features
lr = make_pipeline(PolynomialFeatures(2), LogisticRegression())
lr.fit(X, y)

# print the intercept and coefficients
print(lr.named_steps['logisticregression'].intercept_)
print(lr.named_steps['logisticregression'].coef_)


# plot the decision boundary
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = lr.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
plt.contourf(X, Y, Z, alpha=0.3)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')



# %%
tmp = lr.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
# print the unique values in the prediction
print(np.unique(tmp))
# %%
# plot the sigmoid surface in 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = sigmoid(X + Y)
ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('sigmoid(X1 + X2)')
plt.tight_layout()



# %%
