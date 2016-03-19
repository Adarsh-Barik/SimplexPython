# Author: Adarsh (abarik@purdue.edu)
# This would be the main simplex function
# Features:
# runs in simple tableaue format
# is a dumb function - preprocessing must be done before you call this
# implements Bland's rule to avoid cycling

from DebugPrint import debug
import numpy as np
import sys

# Debug level : 0 - None, 1 - Some, 2 - All (Red) debug statements will be printed

LEVEL = 0
LOCAL_LEVEL = 0


# main simplex routine which runs the simplex algorithm in tableau format
# Input:
# coeff_matrix: coefficient matrix, must have identity matrix
# basic_var_index: basic variables, column index of basic variables
# b: RHS, all positive
# c: cost coefficients
# phase : 1 or 2, need this to remember that Phas 1 contains one extra row for original cost coefficients
# initial_obj_value : initial objective function value
def simplex_solve(coeff_matrix, basic_var_index, b, c, phase, initial_obj_value, should_print_tableau=False):
	# do some sanity testing
	# see if matrix dimensions match before proceeding
	if coeff_matrix.shape[1] is not c.shape[1]:
		print " Coefficient matrix(", coeff_matrix.shape[1], ") and cost coefficient(", c.shape[1], ") must have same number of columns."
		sys.exit()
	if len(coeff_matrix) is not len(b):
		print " Coefficient matrix(", len(coeff_matrix), ") and RHS(", len(b), ") must have same number of rows."
		sys.exit()

	if should_print_tableau:
		LEVEL = 2
	else:
		LEVEL = 0
	# create initial simplex tableau, add cost coefficients and RHS to coefficient matrix
	simplex_tableau = np.append(coeff_matrix, c, 0)
	debug(LOCAL_LEVEL, "Initial Objective Value: ", initial_obj_value)
	b_simplex = np.append(b, np.array([[initial_obj_value]]), 0)
	simplex_tableau = np.append(simplex_tableau, b_simplex, 1)
	debug(LEVEL, "Phase ", phase, " Initial Simplex Tableau:\n", simplex_tableau)

	# assume initial tableau is not optimum
	is_optimum = False
	# get the entering variable, tableau is optimum if we can't get entering variable
	# we use Bland's rule to avoid cycling, check get_entering_variable for more details
	entering_var_index, is_optimum = get_entering_variable(simplex_tableau[-1, :-1])

	# if initial tableau is optimum
	if is_optimum:
		debug(LOCAL_LEVEL, "Optimal achieved and optimum is: ", simplex_tableau[-1, -1])
		return simplex_tableau, basic_var_index

	# start the loop if initial tableau is not optimum
	while not is_optimum:
		# get entering variables column (y) so that we can do minimum ratio test
		# remember to remove row/s for cost coefficients depending on Phase 1 or Phase 2
		y = simplex_tableau[:-1 if phase == 2 else -2, entering_var_index]
		y = y.reshape(len(y), 1)
		debug(LEVEL, "Phase ", phase, " Entering Variable: ", entering_var_index)
		debug(LOCAL_LEVEL, "Phase: ", phase, " Entering Variable: ", entering_var_index, "\nSimplex Tableau:\n", simplex_tableau, "\n y :\n", y)
		# if each element of y is negative then we conclude that problem is unbounded
		# if it happens in Phase 1 then original problem is infeasible
		# if it happens in Phase 2 then we get the extreme direction
		if not (True in (y > 0)):
			if phase == 1:
				print "Problem is infeasible because Phase 1 is unbounded."
				sys.exit()
			elif phase == 2:
				direction = np.zeros((simplex_tableau.shape[1]-1, 1))
				direction[entering_var_index, 0] = 1.
				direction[basic_var_index] = -1*y
				print "Problem is unbounded with one of the direction: \n", direction + 0.
				sys.exit()
		# get current b to do minimum ratio test
		current_b = simplex_tableau[:-1 if phase == 2 else -2, -1].reshape(y.shape)
		# do the ratio test and get leaving variable
		# we use Bland's rule to break the tie, check minimum_ratio_test function for more details
		leaving_var_index, minimum_ratio = minimum_ratio_test(y, current_b, basic_var_index)
		debug(LEVEL, "Phase ", phase, " Leaving Variable: ", basic_var_index[leaving_var_index])
		# update basic variable for next iteration
		basic_var_index[leaving_var_index] = entering_var_index

		# Lets do pivoting and row operations here so that we get simplex tableau for next iteration
		simplex_tableau[leaving_var_index, :] = simplex_tableau[leaving_var_index, :]/simplex_tableau[leaving_var_index, entering_var_index]
		for i in range(len(simplex_tableau)):
			if i != leaving_var_index:
				simplex_tableau[i] = simplex_tableau[i] - simplex_tableau[i, entering_var_index] * simplex_tableau[leaving_var_index]
		debug(LEVEL, "Phase ", phase, " Simplex Tableau: \n", simplex_tableau)
		entering_var_index, is_optimum = get_entering_variable(simplex_tableau[-1, :-1])

	debug(LOCAL_LEVEL, "We have hit optimum folks!\nOptimum Value is: ", simplex_tableau[-1, -1])
	return simplex_tableau, basic_var_index


# function to do minimum ratio test
# input
# y: entering column from simplex tableau
# b: current RHS in simplex tableau
def minimum_ratio_test(y, b, basic_variable):
	# do ratio test only if y>0
	valid_index = np.where(y > np.finfo(np.float32).eps)
	debug(LEVEL, " current b: \n", b)
	debug(LOCAL_LEVEL, "Ratio Test: \n", b[valid_index]/y[valid_index])
	# get the minimum ratio and return it for pivoting
	min_ratio = min(b[valid_index]/y[valid_index])
	# also get leaving variable which has minimum ratio
	# leaving variable will be one with minimum index in case of tie - Bland's rule
	debug(LOCAL_LEVEL, " tied mininimum valid_index ", np.where(b[valid_index]/y[valid_index] == min_ratio))
	debug(LOCAL_LEVEL, " basic_vairable ", basic_variable)
	# Lets get all indexes for tied minimum ratio
	min_ratio_possible_index = valid_index[0][np.where(b[valid_index]/y[valid_index] == min_ratio)[0]]
	debug(LOCAL_LEVEL, "min_ratio_possible_index ", min_ratio_possible_index)
	# take the one which has minimum index in basic variables
	min_ratio_index = basic_variable.index(min(basic_variable[i] for i in min_ratio_possible_index))
	debug(LOCAL_LEVEL, "valid leaving index: ", valid_index, "\nmin_ratio: ", min_ratio, "\nleaving variable index: ", min_ratio_index)
	return min_ratio_index, min_ratio


# function to get entering variable
# input
# c: cost coefficients in simplex tableau
def get_entering_variable(c):
	# get the indices where cost coefficient is greater than 0
	entering_var_index_valid = np.where(c > np.finfo(np.float32).eps)
	debug(LOCAL_LEVEL, " entering_var_index_valid: ", entering_var_index_valid)
	enter_var_index = entering_var_index_valid[0]
	debug(LOCAL_LEVEL, "length of entering_var_index: ", len(enter_var_index))
	# if we don't get any entering variable tell that we are at optimum
	if len(enter_var_index) == 0:
		return '', True
	# return the entering variable index otherwise and tell that we are not yet optimum
	debug(LOCAL_LEVEL, " entering_var_index: ", enter_var_index[0])
	# if len(enter_var_index)>1:
	return enter_var_index[0], False

if __name__ == '__main__':
	A = np.array([[1, 1, 2, 1, 0, 0],
				  [1, 1, 1, 0, 1, 0],
				  [-1, 1, 1, 0, 0, 1]], dtype=float)
	c = np.array([[-1, -1, 4, 0, 0, 0]], dtype=float)
	basic_var = [3, 4, 5]
	b = np.array([[8],
				  [10],
				  [4]], dtype=float)
	ph = 2

	simplex_solve(A, basic_var, b, c, ph, 0)
