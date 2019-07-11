from NN import NeuralNetwork

X = np.linspace(0, 1)
X = X.reshape(1, X.size)

Y = X ** 2
Y = Y.reshape(1, Y.size)

nn_model = NeuralNetwork(
    layer_dims = [100, 100, 1], 
    activations = ['relu', 'relu', 'sigmoid'],
    cost_function = 'l2',
    alpha = 0.1,
    plot_cost = True,
    num_iterations = 4000,
    init_method = 'he'
)

nn_model.fit(X, Y)
Yhat = nn_model.predict(X)

plt.plot(np.squeeze(Yhat))
plt.plot(np.squeeze(Y))
plt.show()

X = np.array([[-4, -4], [-4, 4], [4, -4], [4, 4]]).T
y = np.array([[1], [0], [0], [1]]).T

nn_model = NeuralNetwork(
    layer_dims = [5, 1],
    activations = ['sigmoid', 'sigmoid'], 
    cost_function = 'cross_entropy',
    alpha = 0.1,
    plot_cost = True,
    num_iterations = 4000,
    init_method = 'he'
)

nn_model.fit(X, y)

x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)
z = np.zeros(X.shape)
Z = np.array(z)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = nn_model.predict([[x[i]],[y[j]]])

plt.contourf(X, Y, Z, alpha=.5, cmap='jet_r')
C = plt.contour(X, Y, Z,  colors='black')
plt.plot([4,-4],[4,-4],'x',color='blue')
plt.plot([4,-4],[-4,4],'x',color='red')
plt.axis('equal')
plt.figtext(0.5, 0.01, 'Decision boundary of the fit for our NN.', 
            wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
