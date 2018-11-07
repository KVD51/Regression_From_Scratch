from matplotlib import pyplot as plt


# Define function to find the intercept (b) gradient. Code could be combined with code for m gradient (below)
def get_b_gradient(x, y, m, b):
    diff = 0
    for i in range(len(x)):
        diff += (y[i] - (m * x[i] + b))
    b_gradient = -2 / len(x) * diff
    return b_gradient

# Define function to find the slope (m) gradient
def get_m_gradient(x, y, m, b):
    diff = 0
    for i in range(len(x)):
        diff += x[i] * (y[i] - (m * x[i] + b))
    m_gradient = -2 / len(x) * diff
    return m_gradient

# Define function to find new b and m valies by stepping the gradients along the total error curve using a 'learning rate'. Return new b and m values at the next step
def step_gradient(x, y, m_current, b_current, learning_rate):
    b_gradient = get_b_gradient(x, y, m_current, b_current)
    m_gradient = get_m_gradient(x, y, m_current, b_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [m, b]

# Define function to run step_gradient though a number of iterations until convergence occurs - when b, m (and total loss) changes very little or stops changing. Return b and m values at convergence
def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        m, b = step_gradient(x, y, m, b, learning_rate)
    return [m, b]

# Test the gradient_descent function on some data
months = list(range(1,13))
profit = [250, 380, 320, 490, 610, 540, 550, 630, 780, 720, 790, 850]

m, b = gradient_descent(months, profit, 0.01, 1000)
y = [m * x + b for x in months]

plt.plot(months, profit, "o")
plt.plot(months, y)
plt.show()
