"""
Regularization parameter lambda=0.250

Initializing the network with the following structure (number of neurons per layer): [2 4 3 2]

Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.42000  0.15000  0.40000  
	0.72000  0.10000  0.54000  
	0.01000  0.19000  0.42000  
	0.30000  0.35000  0.68000  

Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.21000  0.67000  0.14000  0.96000  0.87000  
	0.87000  0.42000  0.20000  0.32000  0.89000  
	0.03000  0.56000  0.80000  0.69000  0.09000  

Initial Theta3 (the weights of each neuron, including the bias weight, are stored in the rows):
	0.04000  0.87000  0.42000  0.53000  
	0.17000  0.10000  0.95000  0.69000  


Training set
	Training instance 1
		x: [0.32000   0.68000]
		y: [0.75000   0.98000]
	Training instance 2
		x: [0.83000   0.02000]
		y: [0.75000   0.28000]

--------------------------------------------
Computing the error/cost, J, of the network
	Processing training instance 1
	Forward propagating the input [0.32000   0.68000]
		a1: [1.00000   0.32000   0.68000]

		z2: [0.74000   1.11920   0.35640   0.87440]
		a2: [1.00000   0.67700   0.75384   0.58817   0.70566]

		z3: [1.94769   2.12136   1.48154]
		a3: [1.00000   0.87519   0.89296   0.81480]

		z4: [1.60831   1.66805]
		a4: [0.83318   0.84132]

		f(x): [0.83318   0.84132]
	Predicted output for instance 1: [0.83318   0.84132]
	Expected output for instance 1: [0.75000   0.98000]
	Cost, J, associated with instance 1: 0.791

	Processing training instance 2
	Forward propagating the input [0.83000   0.02000]
		a1: [1.00000   0.83000   0.02000]

		z2: [0.55250   0.81380   0.17610   0.60410]
		a2: [1.00000   0.63472   0.69292   0.54391   0.64659]

		z3: [1.81696   2.02468   1.37327]
		a3: [1.00000   0.86020   0.88336   0.79791]

		z4: [1.58228   1.64577]
		a4: [0.82953   0.83832]

		f(x): [0.82953   0.83832]
	Predicted output for instance 2: [0.82953   0.83832]
	Expected output for instance 2: [0.75000   0.28000]
	Cost, J, associated with instance 2: 1.944

Final (regularized) cost, J, based on the complete training set: 1.90351



--------------------------------------------
Running backpropagation
	Computing gradients based on training instance 1
		delta4: [0.08318   -0.13868]
		delta3: [0.00639   -0.00925   -0.00779]
		delta2: [-0.00087   -0.00133   -0.00053   -0.00070]
		
		Gradients of Theta3 based on training instance 1:
			0.08318  0.07280  0.07427  0.06777  
			-0.13868  -0.12138  -0.12384  -0.11300  

		Gradients of Theta2 based on training instance 1:
			0.00639  0.00433  0.00482  0.00376  0.00451  
			-0.00925  -0.00626  -0.00698  -0.00544  -0.00653  
			-0.00779  -0.00527  -0.00587  -0.00458  -0.00550  

		Gradients of Theta1 based on training instance 1:
			-0.00087  -0.00028  -0.00059  
			-0.00133  -0.00043  -0.00091  
			-0.00053  -0.00017  -0.00036  
			-0.00070  -0.00022  -0.00048  

	Computing gradients based on training instance 2
		delta4: [0.07953   0.55832]
		delta3: [0.01503   0.05809   0.06892]
		delta2: [0.01694   0.01465   0.01999   0.01622]
		
		Gradients of Theta3 based on training instance 2:
			0.07953  0.06841  0.07025  0.06346  
			0.55832  0.48027  0.49320  0.44549  

		Gradients of Theta2 based on training instance 2:
			0.01503  0.00954  0.01042  0.00818  0.00972  
			0.05809  0.03687  0.04025  0.03160  0.03756  
			0.06892  0.04374  0.04775  0.03748  0.04456  

		Gradients of Theta1 based on training instance 2:
			0.01694  0.01406  0.00034  
			0.01465  0.01216  0.00029  
			0.01999  0.01659  0.00040  
			0.01622  0.01346  0.00032  

	The entire training set has been processed. Computing the average (regularized) gradients:
		Final regularized gradients of Theta1:
			0.00804  0.02564  0.04987  
			0.00666  0.01837  0.06719  
			0.00973  0.03196  0.05252  
			0.00776  0.05037  0.08492  

		Final regularized gradients of Theta2:
			0.01071  0.09068  0.02512  0.12597  0.11586  
			0.02442  0.06780  0.04164  0.05308  0.12677  
			0.03056  0.08924  0.12094  0.10270  0.03078  

		Final regularized gradients of Theta3:
			0.08135  0.17935  0.12476  0.13186  
			0.20982  0.19195  0.30343  0.25249  
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from feedforward import Feedforward
from utils import (
    print_row,
    print_initial,
    print_matrix,
    compute_batch,
    compute_forward,
    compute_instance,
)


def verify_example2():
    lambda_ = 0.250

    net = Feedforward([2, 4, 3, 2])

    net.weights[0] = np.array(
        [[0.15000, 0.10000, 0.19000, 0.35000], [0.40000, 0.54000, 0.42000, 0.68000]]
    )
    net.bias[0] = np.array([[0.42000, 0.72000, 0.01000, 0.30000]])

    net.weights[1] = np.array(
        [
            [0.67000, 0.42000, 0.56000],
            [0.14000, 0.20000, 0.80000],
            [0.96000, 0.32000, 0.69000],
            [0.87000, 0.89000, 0.09000],
        ]
    )
    net.bias[1] = np.array([[0.21000, 0.87000, 0.03000]])

    net.weights[2] = np.array(
        [[0.87000, 0.10000], [0.42000, 0.95000], [0.53000, 0.69000]]
    )
    net.bias[2] = np.array([[0.04000, 0.17000]])

    X = np.array([[0.32000, 0.68000], [0.83000, 0.02000]])
    Y = np.array([[0.75000, 0.98000], [0.75000, 0.28000]])

    print()
    print(f"(1). lambda={lambda_:.3f}\n")

    print("(2). Initial Theta1")
    print_matrix(print_initial(net, 0))
    print()

    print("(3). Initial Theta2")
    print_matrix(print_initial(net, 1))
    print()

    print("(4). Initial Theta3")
    print_matrix(print_initial(net, 2))
    print()

    print("(5). Training set")
    for i in range(len(X)):
        print(f"Training instance {i + 1}")
        print(f"x: {print_row(X[i])}")
        print(f"y: {print_row(Y[i])}")
        print()

    instance_costs = []

    for i in range(len(X)):
        x = X[i : i + 1]
        y = Y[i : i + 1]

        a_list, zs, activations = compute_forward(net, x)
        y_pred = activations[-1]
        cost_i = net.compute_loss(y_pred, y, lambda_=0.0)
        instance_costs.append(cost_i)

        print(f"(6.{i + 1}). Training instance {i + 1}")
        print(f"a1: {print_row(a_list[0])}\n")

        print(f"z2: {print_row(zs[0])}")
        print(f"a2: {print_row(a_list[1])}\n")

        print(f"z3: {print_row(zs[1])}")
        print(f"a3: {print_row(a_list[2])}\n")

        print(f"z4: {print_row(zs[2])}")
        print(f"a4: {print_row(a_list[3])}\n")

        print(f"Predicted f(x): {print_row(y_pred)}")
        print(f"Real value {i + 1}: {print_row(y)}")
        print(f"* Cross-entropy J({i + 1}): {cost_i:.3f}\n")

    final_cost = net.compute_loss(net.forward(X)[0], Y, lambda_)
    print(f"* Final (regularized) cost, J: {final_cost:.5f}")

    print()
    print("(7) Backpropagation")

    for i in range(len(X)):
        x = X[i : i + 1]
        y = Y[i : i + 1]

        deltas, grad_thetas = compute_instance(net, x, y)

        print(f"Gradients on instance {i + 1}")
        print(f"delta4: {print_row(deltas[2])}")
        print(f"delta3: {print_row(deltas[1])}")
        print(f"delta2: {print_row(deltas[0])}\n")

        print(f"Gradients of Theta3 based on training instance {i + 1}:")
        print_matrix(grad_thetas[2])
        print()

        print(f"Gradients of Theta2 based on training instance {i + 1}:")
        print_matrix(grad_thetas[1])
        print()

        print(f"Gradients of Theta1 based on training instance {i + 1}:")
        print_matrix(grad_thetas[0])
        print()

    batch_grad_thetas = compute_batch(net, X, Y, lambda_)

    print("Final regularized gradients of Theta1:")
    print_matrix(batch_grad_thetas[0])
    print()

    print("Final regularized gradients of Theta2:")
    print_matrix(batch_grad_thetas[1])
    print()

    print("Final regularized gradients of Theta3:")
    print_matrix(batch_grad_thetas[2])
    print()


if __name__ == "__main__":
    verify_example2()

"""
Result:

(1). lambda=0.250

(2). Initial Theta1
0.42000  0.15000  0.40000
0.72000  0.10000  0.54000
0.01000  0.19000  0.42000
0.30000  0.35000  0.68000

(3). Initial Theta2
0.21000  0.67000  0.14000  0.96000  0.87000
0.87000  0.42000  0.20000  0.32000  0.89000
0.03000  0.56000  0.80000  0.69000  0.09000

(4). Initial Theta3
0.04000  0.87000  0.42000  0.53000
0.17000  0.10000  0.95000  0.69000

(5). Training set
Training instance 1
x: [0.32000   0.68000]
y: [0.75000   0.98000]

Training instance 2
x: [0.83000   0.02000]
y: [0.75000   0.28000]

(6.1). Training instance 1
a1: [1.00000   0.32000   0.68000]

z2: [0.74000   1.11920   0.35640   0.87440]
a2: [1.00000   0.67700   0.75384   0.58817   0.70566]

z3: [1.94769   2.12136   1.48154]
a3: [1.00000   0.87519   0.89296   0.81480]

z4: [1.60831   1.66805]
a4: [0.83318   0.84132]

Predicted f(x): [0.83318   0.84132]
Real value 1: [0.75000   0.98000]
* Cross-entropy J(1): 0.791

(6.2). Training instance 2
a1: [1.00000   0.83000   0.02000]

z2: [0.55250   0.81380   0.17610   0.60410]
a2: [1.00000   0.63472   0.69292   0.54391   0.64659]

z3: [1.81696   2.02468   1.37327]
a3: [1.00000   0.86020   0.88336   0.79791]

z4: [1.58228   1.64577]
a4: [0.82953   0.83832]

Predicted f(x): [0.82953   0.83832]
Real value 2: [0.75000   0.28000]
* Cross-entropy J(2): 1.944

* Final (regularized) cost, J: 1.90351

(7) Backpropagation
Gradients on instance 1
delta4: [0.08318   -0.13868]
delta3: [0.00639   -0.00925   -0.00779]
delta2: [-0.00087   -0.00133   -0.00053   -0.00070]

Gradients of Theta3 based on training instance 1:
0.08318  0.07280  0.07427  0.06777
-0.13868  -0.12138  -0.12384  -0.11300

Gradients of Theta2 based on training instance 1:
0.00639  0.00433  0.00482  0.00376  0.00451
-0.00925  -0.00626  -0.00698  -0.00544  -0.00653
-0.00779  -0.00527  -0.00587  -0.00458  -0.00550

Gradients of Theta1 based on training instance 1:
-0.00087  -0.00028  -0.00059
-0.00133  -0.00043  -0.00091
-0.00053  -0.00017  -0.00036
-0.00070  -0.00022  -0.00048

Gradients on instance 2
delta4: [0.07953   0.55832]
delta3: [0.01503   0.05809   0.06892]
delta2: [0.01694   0.01465   0.01999   0.01622]

Gradients of Theta3 based on training instance 2:
0.07953  0.06841  0.07025  0.06346
0.55832  0.48027  0.49320  0.44549

Gradients of Theta2 based on training instance 2:
0.01503  0.00954  0.01042  0.00818  0.00972
0.05809  0.03687  0.04025  0.03160  0.03756
0.06892  0.04374  0.04775  0.03748  0.04456

Gradients of Theta1 based on training instance 2:
0.01694  0.01406  0.00034
0.01465  0.01216  0.00029
0.01999  0.01659  0.00040
0.01622  0.01346  0.00032

Final regularized gradients of Theta1:
0.00804  0.02564  0.04987
0.00666  0.01837  0.06719
0.00973  0.03196  0.05252
0.00776  0.05037  0.08492

Final regularized gradients of Theta2:
0.01071  0.09068  0.02512  0.12597  0.11586
0.02442  0.06780  0.04164  0.05308  0.12677
0.03056  0.08924  0.12094  0.10270  0.03078

Final regularized gradients of Theta3:
0.08135  0.17935  0.12476  0.13186
0.20982  0.19195  0.30343  0.25249
"""