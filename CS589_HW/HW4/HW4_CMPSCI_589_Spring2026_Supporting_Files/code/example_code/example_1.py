"""
Regularization parameter lambda=0.000

Initializing the network with the following structure (number of neurons per layer): [1 2 1]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.40000  0.10000  
	0.30000  0.20000  


Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.70000  0.50000  0.60000  


Training set
	Training instance 1
		x: [0.13000]
		y: [0.90000]
	Training instance 2
		x: [0.42000]
		y: [0.23000]

--------------------------------------------
Computing the error/cost, J, of the network
	Processing training instance 1
	Forward propagating the input [0.13000]
		a1: [1.00000   0.13000]

		z2: [0.41300   0.32600]
		a2: [1.00000   0.60181   0.58079]

		z3: [1.34937]
		a3: [0.79403]

		f(x): [0.79403]
	Predicted output for instance 1: [0.79403]
	Expected output for instance 1: [0.90000]
	Cost, J, associated with instance 1: 0.366

	Processing training instance 2
	Forward propagating the input [0.42000]
		a1: [1.00000   0.42000]

		z2: [0.44200   0.38400]
		a2: [1.00000   0.60874   0.59484]

		z3: [1.36127]
		a3: [0.79597]

		f(x): [0.79597]
	Predicted output for instance 2: [0.79597]
	Expected output for instance 2: [0.23000]
	Cost, J, associated with instance 2: 1.276

Final (regularized) cost, J, based on the complete training set: 0.82098



--------------------------------------------
Running backpropagation
	Computing gradients based on training instance 1
		delta3: [-0.10597]
		delta2: [-0.01270   -0.01548]
		
		Gradients of Theta2 based on training instance 1:
			-0.10597  -0.06378  -0.06155  

		Gradients of Theta1 based on training instance 1:
			-0.01270  -0.00165  
			-0.01548  -0.00201  

	Computing gradients based on training instance 2
		delta3: [0.56597]
		delta2: [0.06740   0.08184]
		
		Gradients of Theta2 based on training instance 2:
			0.56597  0.34452  0.33666  

		Gradients of Theta1 based on training instance 2:
			0.06740  0.02831  
			0.08184  0.03437  

	The entire training set has been processed. Computing the average (regularized) gradients:
		Final regularized gradients of Theta1:
			0.02735  0.01333  
			0.03318  0.01618  

		Final regularized gradients of Theta2:
			0.23000  0.14037  0.13756  
"""
import os
import sys
from typing_extensions import Final

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from feedforward import Feedforward
from utils import (
    print_row,
    print_matrix,
    print_initial,
    compute_forward,
    compute_instance,
    compute_batch,
)


def verify_example1():
    lambda_ = 0.000

    net = Feedforward([1, 2, 1])

    net.weights[0] = np.array([[0.10000, 0.20000]])
    net.bias[0] = np.array([[0.40000, 0.30000]])
    net.weights[1] = np.array([[0.50000], [0.60000]])
    net.bias[1] = np.array([[0.70000]])

    X = np.array([[0.13000], [0.42000]])
    Y = np.array([[0.90000], [0.23000]])

    print()
    print(f"Regularization parameter lambda={lambda_:.3f}\n")
    print(f"Initializing the network with the following structure (number of neurons per layer): [1 2 1]\n")

    print("Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):")
    print_matrix(print_initial(net, 0))
    print()

    print("Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):")
    print_matrix(print_initial(net, 1))
    print()

    print("Training set")
    for i in range(len(X)):
        print(f"Training instance {i + 1}")
        print(f"x: {print_row(X[i])}")
        print(f"y: {print_row(Y[i])}")

    print()
    print("--------------------------------------------")
    print("Computing the error/cost, J, of the network")
    instance_costs = []

    for i in range(len(X)):
        print(f"Processing training instance {i + 1}")
        x = X[i : i + 1]
        y = Y[i : i + 1]

        print(f"Forward propagating the input {print_row(x)}")

        a_list, zs, activations = compute_forward(net, x)
        y_pred = activations[-1]
        cost_i = net.compute_loss(y_pred, y, lambda_=0.0)
        instance_costs.append(cost_i)

        print(f"a1: {print_row(a_list[0])}\n")

        print(f"z2: {print_row(zs[0])}")
        print(f"a2: {print_row(a_list[1])}\n")

        print(f"z3: {print_row(zs[1])}")
        print(f"a3: {print_row(a_list[2])}\n")

        print(f"Predicted output for instance {i + 1}: {print_row(y_pred)}")
        print(f"Expected output for instance {i + 1}: {print_row(y)}")
        print(f"Cost, J, associated with instance {i + 1}: {cost_i:.3f}")
        print()

    final_cost = net.compute_loss(net.forward(X)[0], Y, lambda_)
    print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}")

    print()
    print("--------------------------------------------")
    print("Running backpropagation")

    for i in range(len(X)):
        x = X[i : i + 1]
        y = Y[i : i + 1]

        deltas, grad_thetas = compute_instance(net, x, y)

        print(f"Computing gradients based on training instance {i + 1}")
        print(f"delta3: {print_row(deltas[1])}")
        print(f"delta2: {print_row(deltas[0])}\n")

        print(f"Gradients of Theta2 based on training instance {i + 1}:")
        print_matrix(grad_thetas[1])
        print()

        print(f"Gradients of Theta1 based on training instance {i + 1}:")
        print_matrix(grad_thetas[0])
        print()

    print("The entire training set has been processed. Computing the average (regularized) gradients:")
    batch_grad_thetas = compute_batch(net, X, Y, lambda_)

    print("Final regularized gradients of Theta1:")
    print_matrix(batch_grad_thetas[0])
    print()

    print("Final regularized gradients of Theta2:")
    print_matrix(batch_grad_thetas[1])
    print()


if __name__ == "__main__":
    verify_example1()

"""
Result:

Regularization parameter lambda=0.000

Initializing the network with the following structure (number of neurons per layer): [1 2 1]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
0.40000  0.10000
0.30000  0.20000

Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
0.70000  0.50000  0.60000

Training set
Training instance 1
x: [0.13000]
y: [0.90000]
Training instance 2
x: [0.42000]
y: [0.23000]

--------------------------------------------
Computing the error/cost, J, of the network
Processing training instance 1
Forward propagating the input [0.13000]
a1: [1.00000   0.13000]

z2: [0.41300   0.32600]
a2: [1.00000   0.60181   0.58079]

z3: [1.34937]
a3: [0.79403]

Predicted output for instance 1: [0.79403]
Expected output for instance 1: [0.90000]
Cost, J, associated with instance 1: 0.366

Processing training instance 2
Forward propagating the input [0.42000]
a1: [1.00000   0.42000]

z2: [0.44200   0.38400]
a2: [1.00000   0.60874   0.59484]

z3: [1.36127]
a3: [0.79597]

Predicted output for instance 2: [0.79597]
Expected output for instance 2: [0.23000]
Cost, J, associated with instance 2: 1.276

Final (regularized) cost, J, based on the complete training set: 0.82098

--------------------------------------------
Running backpropagation
Computing gradients based on training instance 1
delta3: [-0.10597]
delta2: [-0.01270   -0.01548]

Gradients of Theta2 based on training instance 1:
-0.10597  -0.06378  -0.06155

Gradients of Theta1 based on training instance 1:
-0.01270  -0.00165
-0.01548  -0.00201

Computing gradients based on training instance 2
delta3: [0.56597]
delta2: [0.06740   0.08184]

Gradients of Theta2 based on training instance 2:
0.56597  0.34452  0.33666

Gradients of Theta1 based on training instance 2:
0.06740  0.02831
0.08184  0.03437

The entire training set has been processed. Computing the average (regularized) gradients:
Final regularized gradients of Theta1:
0.02735  0.01333
0.03318  0.01618

Final regularized gradients of Theta2:
0.23000  0.14037  0.13756
"""