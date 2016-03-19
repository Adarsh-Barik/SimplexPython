# Author : Adarsh (abarik@purdue.edu)
# Main stem program that calls others to implement Two Phase Simplex method
# "You either die infeasible or live long enough to become optimal." - Two Phase
# I know there is also unbounded but that is optimal with optimal value +/- infinity


import ConfigParser
from DebugPrint import debug
import sys
from Preprocessing import PreProcessing
from SimplexMain import simplex_solve
import numpy as np

# Debug level : 0 - None, 1 - Some, 2 - All (Red) debug statements will be printed
LEVEL = 0
LOCAL_LEVEL = 0

# Config file
# Searches for file named "config" in working directory
# Reads configuration from there
# Read the file named "config" in to understand each configuration
config = ConfigParser.ConfigParser()
if(len(config.read('config')) == 0):
	print "config file is missing from current directory."
	sys.exit()
debug(LOCAL_LEVEL, config.sections())

# Which input method to choose
use_human_readable_config = 'True' in config.get('DEFAULT_CONFIG', 'use_human_readable_config')
use_standard_form_config = 'True' in config.get('DEFAULT_CONFIG', 'use_standard_form_config')

should_print_tableau = 'True' in config.get('DEFAULT_CONFIG', 'should_print_tableau')

if use_human_readable_config and use_standard_form_config:
	print "Can't use both configuration simultaneously."
	sys.exit()

if not use_human_readable_config and not use_standard_form_config:
	print "Choose atleast one input method."
	sys.exit()

# If human readable input method is chosen
if use_human_readable_config:
	config_section = 'HUMAN_READABLE'
	# get the problem_file path from configuration
	problem_file = config.get(config_section, 'problem_file')
	# should we assume variables to be non-negative
	assume_non_negative_variable = 'True' in config.get(config_section, 'assume_non_negative_variable')
	# read the problem_file, ignore comments obviously (starts with #)
	problem = ''
	try:
		for line in open(problem_file):
			li = line.strip()
			if not li.startswith('#') and li:
				problem += li.rstrip() + '\t'
	except IOError as e:
		print "{0} {1} {2}".format(e.errno, e.strerror, e.filename)
		sys.exit()

	problem = problem.replace(',', '\t')
	problem = problem.replace(';', '\t')
	debug(LEVEL, "PROBLEM: \n", problem)

	# Lets start pre-processing the problem now (its in human readable format, can't use it as it is)
	pre_processed_problem = PreProcessing(problem)
	# get the variables
	pre_processed_problem.parse_variables()
	# if variables are not non-negative then get their signs
	# we'll use this to change the problem to standard format
	if not assume_non_negative_variable:
		pre_processed_problem.get_variable_sign()
	# get cost coefficients
	pre_processed_problem.get_original_cost_coeff()
	# get coefficient matrix
	pre_processed_problem.get_coeff_matrix()
	# standardize everything
	# also figure out if we need Two Phase (all <= constraints and we are a happy bloke)
	is_two_phase_required, standard_coeff_matrix, standard_b = pre_processed_problem.get_standardized_coeff_matrix(pre_processed_problem.constraints, pre_processed_problem.coeff_matrix, pre_processed_problem.b)

	# if we require Two Phase
	if is_two_phase_required:
		# This means that we'll be adding artificial variables to original problem and will test feasibility
		# Lets get coefficient matrix and cost coefficients for Phase 1 problem
		artificial_coeff_matrix = np.append(standard_coeff_matrix, np.eye(len(standard_coeff_matrix)), 1)
		artificial_cost_coeff = np.zeros((1, artificial_coeff_matrix.shape[1]))
		# cost coefficient for Phase 1 for non-basic variables
		for i in range(standard_coeff_matrix.shape[1]):
			artificial_cost_coeff[0, i] = artificial_coeff_matrix[:, i].sum()

		# Lets also calculate cost coefficient for original problem
		# we'll keep this as one of the row in Phase 1 coefficient matrix
		# this would save us time later in Phase 2, cunning isn't it?
		number_of_extra_var = artificial_coeff_matrix.shape[1] - pre_processed_problem.coeff_matrix.shape[1]
		extra_cost_coeff = np.zeros((1, number_of_extra_var))
		standard_cost_coeff = np.append(pre_processed_problem.cost_coeff, extra_cost_coeff).reshape(1, artificial_coeff_matrix.shape[1])
		artificial_coeff_matrix = np.append(artificial_coeff_matrix, standard_cost_coeff, 0)

		# we'll also need indices of basic variables to start the tableau
		# these will be artificial variables
		basic_var = range(standard_coeff_matrix.shape[1], artificial_coeff_matrix.shape[1])

		# add a zero to RHS for the row corresponding to original cost coefficient
		# again we will use this row in Phase 2
		b = np.append(standard_b, np.array([[0.]]), 0)

		# Now that everything is set, lets call our simplex routine (Read how it works in SimplexMain.py)
		# initial objective function value is obviously summation of intial b (RHS), e.g. a1 + a2 + a3..
		phase1_final_tableau, phase1_final_variables = simplex_solve(artificial_coeff_matrix, basic_var, b, artificial_cost_coeff, 1, b.sum(), should_print_tableau)

		# if objective function value at the end of Phase 1 is not zero then problem is not feasible - stop the program here
		# comparison of float with exact zero is tricky, we'll be using a comparison with machine epsilon
		# here machine epsilon is 1.1920929e-07
		if np.abs(phase1_final_tableau[-1, -1]) >= np.finfo(np.float32).eps:
			print "Optimal value is not zero at the end of Phase 1, hence problem is infeasible."
			sys.exit()

		# check if there any artificial variable in the final solution of Phase 1
		# if the are there then we'll remove them before we move to Phase 2
		artificial_variable_index = range(standard_coeff_matrix.shape[1], artificial_coeff_matrix.shape[1])
		artificial_var_in_basis = list(set(artificial_variable_index).intersection(phase1_final_variables))
		debug(LOCAL_LEVEL, "artificial_var_in_basis: ", artificial_var_in_basis)
		if len(artificial_var_in_basis):
			# any artificial variable that is in basis will have a value of zero
			# one way to remove it is to make it a leaving variable and enter a variable which is not artificial
			# and not in the current basis -- also called legitimate nonbasic variable
			# condition 1 : now the above method works if pivot element corresponding to artificial variable and legitimate nonbasic
			# variable is non-zero and artificial variable leaves the basis
			# condition 2 : if this pivot element is zero for all available legitimate nonbasic variable then this
			# implies that corresponding constraint is redundant and plays no role whatsoever in phase 2 which is equivalent
			# to saying that coefficient matrix associated with the standard form (NOT canonical form) of the LP doesn't have full row
			# rank, we can keep this constraint or remove it, it doesn't affect our final solution
			# lets implement this now
			print "We have found artificial variable/s in basis after Phase 1. It will be removed."
			# get legitimate nonbasic variables
			non_artificial_variable_index = range(0, standard_coeff_matrix.shape[1])
			legitimate_nonbasic_variable = list(set(non_artificial_variable_index).difference(phase1_final_variables))
			artificial_var_in_basis_index = []
			for i in artificial_var_in_basis:
					artificial_var_in_basis_index.append(phase1_final_variables.index(i))
			# we'll also catch the redundant constraints
			redundant_constraint_index = []
			for i in artificial_var_in_basis_index:
				# this would be our row for leaving variable
				artificial_matrix = phase1_final_tableau[i, legitimate_nonbasic_variable]
				debug(LOCAL_LEVEL, "artificial_matrix for constraint ", i+1, " is \n", artificial_matrix)
				# check if pivoting is possible
				# if not possible then constraint is redundant and can be removed - condition 2
				if not (False in (abs(artificial_matrix) < np.finfo(np.float32).eps)):
					print "Constraint: ", pre_processed_problem.constraints[i], "is redundant."
					# don't remove anything yet but lets catch the constraint index that we can remove
					redundant_constraint_index.append(i)
				# if pivoting is possible then do pivoting and remove the artificial variable from basis
				else:
					pivot_column = np.where(abs(artificial_matrix) > np.finfo(np.float32).eps)[0][0]
					# Lets do pivoting at any non-zero element and remove the artificial variable from basis
					phase1_final_tableau[i, :] = phase1_final_tableau[i, :]/phase1_final_tableau[i, legitimate_nonbasic_variable[pivot_column]]
					for j in range(len(phase1_final_tableau)):
						if j != i:
							phase1_final_tableau[j] = phase1_final_tableau[j] - phase1_final_tableau[j, legitimate_nonbasic_variable[pivot_column]]*phase1_final_tableau[i]
					phase1_final_variables[i] = legitimate_nonbasic_variable[pivot_column]
			# we can remove redundant constraints now (we can keep them as well and they won't affect anything)
			# however we'll remove them as we are asked to in programming requirements 3
			redundant_constraint_index.sort()
			redundant_constraint_index.reverse()
			for i in redundant_constraint_index:
				phase1_final_tableau = np.delete(phase1_final_tableau, i, 0)
				phase1_final_variables.remove(phase1_final_variables[i])

		# Lets run Phase 2 now
		# Take the necessary elements of the final tableau of Phase 1 (i.e. eliminate the artificial variables column)
		# Remember! we carried cost coefficient of original problem in one of the row of coefficient matrix,
		# that gives us cost coefficient and initial objective function value for Phase 2, Sweet!
		b_phase2 = phase1_final_tableau[:standard_coeff_matrix.shape[0], -1].reshape(standard_coeff_matrix.shape[0], 1)
		c_phase2 = phase1_final_tableau[-2, :standard_coeff_matrix.shape[1]].reshape(1, standard_coeff_matrix.shape[1])

		# call simplex routine for Phase 2 (unboundedness is handled internally by this routine)
		final_tableau, final_variables = simplex_solve(phase1_final_tableau[:standard_coeff_matrix.shape[0], :standard_coeff_matrix.shape[1]], phase1_final_variables, b_phase2, c_phase2, 2, phase1_final_tableau[-2, -1], should_print_tableau)

		# Display the final optimal result
		print "OPTIMAL VALUE\t", final_tableau[-1, -1] if pre_processed_problem.is_min_problem else -1 * final_tableau[-1, -1]

		# Display values for variables and slack
		if assume_non_negative_variable:
			original_var = pre_processed_problem.original_variable
			j = -1
			for i in final_variables:
				j += 1
				if i < len(original_var):
					print original_var[i], "\t", final_tableau[j, -1]
				else:
					print "Variable ", i+1, "\t", final_tableau[j, -1], "\t<-- Slack Variable"
		else:
			signed_var = pre_processed_problem.signed_variable
			j = -1
			for i in final_variables:
				j += 1
				if i < len(signed_var):
					print signed_var[i], "\t", final_tableau[j, -1]
				else:
					print "Variable ", i+1, "\t", final_tableau[j, -1], "\t<-- Slack Variable"
			for i in pre_processed_problem.negative_variables:
				print i, "\t", "-1 *", i, "-"
			for i in pre_processed_problem.free_variables:
				print i, "\t", i, '+ - ', i, '-'

	else:
		# If we don't need Two Phase then single call to Simplex does the trick
		# get cost coefficient for slack variables (0)
		number_of_slack = standard_coeff_matrix.shape[1] - pre_processed_problem.coeff_matrix.shape[1]
		slack_cost_coeff = np.zeros((1, number_of_slack))
		standard_cost_coeff = np.append(pre_processed_problem.cost_coeff, slack_cost_coeff).reshape(1, standard_coeff_matrix.shape[1])
		# starting basic variables will be the slack variables
		basic_var = range(pre_processed_problem.coeff_matrix.shape[1], standard_coeff_matrix.shape[1])

		# Call simplex routine and enjoy life (unboundedness is handelled internally by this routine)
		final_tableau, final_variables = simplex_solve(standard_coeff_matrix, basic_var, standard_b, standard_cost_coeff, 2, 0, should_print_tableau)

		# Display results (Optimal value and variables)
		print "OPTIMAL VALUE\t", final_tableau[-1, -1] if pre_processed_problem.is_min_problem else -1 * final_tableau[-1, -1]

		if assume_non_negative_variable:
			original_var = pre_processed_problem.original_variable
			j = -1
			for i in final_variables:
				j += 1
				if i < len(original_var):
					print original_var[i], "\t", final_tableau[j, -1]
				else:
					print "Variable ", i+1, "\t", final_tableau[j, -1], "\t<-- Slack Variable"
		else:
			signed_var = pre_processed_problem.signed_variable
			j = -1
			for i in final_variables:
				j += 1
				if i < len(signed_var):
					print signed_var[i], "\t", final_tableau[j, -1]
				else:
					print "Variable ", i+1, "\t", final_tableau[j, -1], "\t<-- Slack Variable"
			for i in pre_processed_problem.negative_variables:
				print i, "\t", "-1 *", i, "-"
			for i in pre_processed_problem.free_variables:
				print i, "\t", i, '+ - ', i, '-'
# Now if input method is in Standard Form
elif use_standard_form_config:
	# get the path for coeff_matrix_file, cost_coeff_file and b_file from "config"
	# read "config" for more details
	coeff_matrix_file = config.get('STANDARD_FORM', 'coeff_matrix_file')
	cost_coeff_file = config.get('STANDARD_FORM', 'cost_coeff_file')
	b_file = config.get('STANDARD_FORM', 'b_file')
	# Check if its a minimization problem
	is_min_problem = 'True' in config.get('STANDARD_FORM', 'is_min_problem')

	# Load coefficient matrix, cost coefficients and b as array from files
	try:
		standard_coeff_matrix = np.loadtxt(coeff_matrix_file, dtype=float)
		cost_coeff = -1 * np.loadtxt(cost_coeff_file, dtype=float)
		standard_b = np.loadtxt(b_file, dtype=float)
	except IOError as e:
		print "{0} {1} {2}".format(e.errno, e.strerror, e.filename)

	standard_b = standard_b.reshape(standard_b.shape[0], 1)

	# Add artificial variables to standard matrix to start phase 1
	# Now on few good cases we don't need this but determining those few cases are not worth the effort
	artificial_coeff_matrix = np.append(standard_coeff_matrix, np.eye(len(standard_coeff_matrix)), 1)
	# as before lets get cost coefficients, coefficient matrix and b for Phase 1
	# again as before we'll also carry cost coefficient for original problem so that we can use it later
	artificial_cost_coeff = np.zeros((1, artificial_coeff_matrix.shape[1]))
	for i in range(standard_coeff_matrix.shape[1]):
		artificial_cost_coeff[0, i] = artificial_coeff_matrix[:, i].sum()
	number_of_extra_var = artificial_coeff_matrix.shape[1] - standard_coeff_matrix.shape[1]
	standard_cost_coeff = np.append(cost_coeff, np.zeros((1, number_of_extra_var))).reshape(1, artificial_coeff_matrix.shape[1])
	artificial_coeff_matrix = np.append(artificial_coeff_matrix, standard_cost_coeff, 0)
	basic_var = range(standard_coeff_matrix.shape[1], artificial_coeff_matrix.shape[1])
	b = np.append(standard_b, np.array([[0.]]), 0)
	# Lets call simplex routine for Phase 1
	phase1_final_tableau, phase1_final_variables = simplex_solve(artificial_coeff_matrix, basic_var, b, artificial_cost_coeff, 1, b.sum(), should_print_tableau)

	# if Phase 1 optimal value is not 0 then problem is infeasible
	if abs(phase1_final_tableau[-1, -1]) >= np.finfo(np.float32).eps:
		print "Optimal value is not zero at the end of Phase 1, hence problem is infeasible."
		sys.exit()

	# check if there any artificial variable in the final solution of Phase 1
	# if the are there then we'll remove them before we move to Phase 2
	artificial_variable_index = range(standard_coeff_matrix.shape[1], artificial_coeff_matrix.shape[1])
	artificial_var_in_basis = list(set(artificial_variable_index).intersection(phase1_final_variables))
	debug(LEVEL, "artificial_var_in_basis: ", artificial_var_in_basis)
	if len(artificial_var_in_basis):
		# any artificial variable that is in basis will have a value of zero
		# one way to remove it is to make it a leaving variable and enter a variable which is not artificial
		# and not in the current basis -- also called legitimate nonbasic variable
		# condition 1 : now the above method works if pivot element corresponding to artificial variable and legitimate nonbasic
		# variable is non-zero and artificial variable leaves the basis
		# condition 2 : if this pivot element is zero for all available legitimate nonbasic variable then this
		# implies that corresponding constraint is redundant and plays no role whatsoever in phase 2 which is equivalent
		# to saying that coefficient matrix associated with the standard form (NOT canonical form) of the LP doesn't have full row
		# rank, we can keep this constraint or remove it, it doesn't affect our final solution
		# lets implement this now
		print "We have found artificial variable/s in basis after Phase 1. It will be removed."

		# get legitimate nonbasic variables
		non_artificial_variable_index = range(0, standard_coeff_matrix.shape[1])
		legitimate_nonbasic_variable = list(set(non_artificial_variable_index).difference(phase1_final_variables))
		artificial_var_in_basis_index = []
		for i in artificial_var_in_basis:
				artificial_var_in_basis_index.append(phase1_final_variables.index(i))
		# we'll also catch the redundant constraints
		redundant_constraint_index = []
		for i in artificial_var_in_basis_index:
			# this would be our row for leaving variable
			artificial_matrix = phase1_final_tableau[i, legitimate_nonbasic_variable]
			debug(LOCAL_LEVEL, "artificial_matrix for constraint ", i+1, " is \n", artificial_matrix)
			# check if pivoting is possible
			# if not possible then constraint is redundant and can be removed - condition 2
			if not (False in (abs(artificial_matrix) < np.finfo(np.float32).eps)):
				# don't remove anything yet but lets catch the constraint index that we can remove
				redundant_constraint_index.append(i)
			# if pivoting is possible then do pivoting and remove the artificial variable from basis
			else:
				pivot_column = np.where(abs(artificial_matrix) > np.finfo(np.float32).eps)[0][0]
				# Lets do pivoting at any non-zero element and remove the artificial variable from basis
				phase1_final_tableau[i, :] = phase1_final_tableau[i, :]/phase1_final_tableau[i, legitimate_nonbasic_variable[pivot_column]]
				for j in range(len(phase1_final_tableau)):
					if j != i:
						phase1_final_tableau[j] = phase1_final_tableau[j] - phase1_final_tableau[j, legitimate_nonbasic_variable[pivot_column]]*phase1_final_tableau[i]
				phase1_final_variables[i] = legitimate_nonbasic_variable[pivot_column]
		# we can remove redundant constraints now (we can keep them as well and they won't affect anything)
		# however we'll remove them as we are asked to in programming requirements 3
		redundant_constraint_index.sort()
		redundant_constraint_index.reverse()
		for i in redundant_constraint_index:
			phase1_final_tableau = np.delete(phase1_final_tableau, i, 0)
			phase1_final_variables.remove(phase1_final_variables[i])

	# Lets run Phase 2 now
	b_phase2 = phase1_final_tableau[:standard_coeff_matrix.shape[0], -1].reshape(standard_coeff_matrix.shape[0], 1)
	c_phase2 = phase1_final_tableau[-2, :standard_coeff_matrix.shape[1]].reshape(1, standard_coeff_matrix.shape[1])
	final_tableau, final_variables = simplex_solve(phase1_final_tableau[:standard_coeff_matrix.shape[0], :standard_coeff_matrix.shape[1]], phase1_final_variables, b_phase2, c_phase2, 2, phase1_final_tableau[-2, -1], should_print_tableau)

	print "OPTIMAL VALUE\t", final_tableau[-1, -1] if is_min_problem else -1 * final_tableau[-1, -1]
	j = -1
	for i in final_variables:
		j += 1
		print "Variable no. ", i, "\t", final_tableau[j, -1]
