#Custom Training: Basics
import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

#Variables
#Using python state
x = tf.zeros([10, 10])
x += 2 #This is equivalent to x = x + 2, which does not mutate the original value of x
print(x)

v = tfe.Variable(1.0)
assert v.numpy() == 1.0

#Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

#Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0

#Example: Fitting a linear model
#1. Define the model
#2. Define a loss function
#3. Obtain training data
#4. Run through the training data and use an "optimizer" to adjust the variables to fit the data

#Define the Model
class Model(object):
    def __init__(self):
        self.W = tfe.Variable(5.0)
        self.b = tfe.Variable(0.0)
    def __call__(self, x):
        return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0

#Define a loss function: using standard L2 loss
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

#Obtain training data
#Synthesize the data with some noise
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape = [NUM_EXAMPLES])
noise = tf.random_normal(shape = [NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

print('Current loss: '),
print(loss(model(inputs), outputs).numpy())

#Define a training loop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

#Finally let's repeatedly run through the training data and see how W and b evolve
model = Model()

#collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    train(model, inputs, outputs, learning_rate = 0.1)
    print('Epoch %2d: W = %1.2f, b = %1.2f, loss = %2.5f' %(epoch, Ws[-1], bs[-1], current_loss))

#Let's plot it all
plt.plot(epochs, Ws, 'r', epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()
