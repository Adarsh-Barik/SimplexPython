# author : Adarsh (abarik@purdue.edu)
# This file is to process input if its in human readable format
# and convert those input in a usable form for main Simplex program


import re
from DebugPrint import debug
import numpy as np
from fractions import Fraction

# Debug level : 0 - None, 1 - Some, 2 - All (Red) debug statements will be printed
LEVEL = 0
LOCAL_LEVEL = 0


class PreProcessing():
	"""class to preprocess input"""

	# get the problem statement and start preprocessing
	def __init__(self, *args, **kwargs):
		# if inputting as a function
		self.problem = args[0]

		# Lets replace some commonly used strings in problem statement
		# to a standard string

		# min, Min, Minimize, MINIMIZE etc -> MIN
		replace_mins = re.compile('min\.|minimize|min', re.IGNORECASE)
		self.problem = replace_mins.sub('MIN', self.problem)

		# max, Max, Maximize, MAXIMIZE etc -> MAX
		replace_maxs = re.compile('max\.|maximize|max', re.IGNORECASE)
		self.problem = replace_maxs.sub('MAX', self.problem)

		# subject to, such that, sub. to, s.t., sub to etc -> SUB TO
		replace_sub_to = re.compile('s\.t\.|sub\.|sub to|subject to|such that|st', re.IGNORECASE)
		self.problem = replace_sub_to.sub('SUB TO', self.problem)

		self.original_variable = []
		debug(LOCAL_LEVEL, 'PROBLEM: ', self.problem)
		# get the constraints from problem separately in raw format
		self.constraints = filter(None, self.problem.replace('SUB TO', '\t').split('\t'))[1:]
		debug(LOCAL_LEVEL, "constraints: ", self.constraints)
		# by default assume non-negative variables, this will be changed later if needed
		self.variable_sign_needed = False

	# get variables from problem statement
	def parse_variables(self):
		# figure out how many variables are there and get all of them
		problem_str = "+".join(self.problem.split('\\'))
		# remove unnecessary words, lets focus on variables
		problem_split = ' '.join(re.split('\+|\-|MIN|SUB TO|=|<=|>=', problem_str)).replace('MIN', '').replace('MAX', '').replace('SUB TO', '').strip()
		# get variables : they can be of form x, x1, X, X1
		self.original_variable = re.findall('([A-Z]\\b|[A-Z][0-9]*|[a-z]\\b|[a-z][0-9]*\\b)', problem_split)
		# remove any duplicate entry and sort them alphabatically - this doesn't matter but looks good
		self.original_variable = list(set(filter(None, self.original_variable)))
		self.original_variable.sort()
		debug(LEVEL, "Original variables: ", self.original_variable)

	# get cost coefficient from objective function
	def get_original_cost_coeff(self):
		# get cost coefficient in an array
		# the array would correspond to original variables - makes sense!
		# we'll worry about slacks while we input them to main simplex routine, they'll be 0 anyway
		self.cost_coeff = np.zeros((1, len(self.original_variable)))
		# get objective function
		self.obj_function = self.problem.strip().split('\t')[0]
		debug(LEVEL, "Objective function: ", self.obj_function)
		# don't need unencessary words, chuck them out
		obj_function_strip = self.obj_function.replace('MIN', '').replace('MAX', '').strip()
		# get coefficients using get_coeff function - check it out for detail
		variable_index = 0
		for variable in self.original_variable:
			self.cost_coeff[0, variable_index] = -1 * self.get_coeff(obj_function_strip, variable)
			debug(LOCAL_LEVEL, "variable_index ", variable_index)
			variable_index = variable_index+1

		# if problem is maximization then cost coefficients will be negative
		self.is_min_problem = "MIN" in self.obj_function
		if self.is_min_problem is False:
			self.cost_coeff = -1. * self.cost_coeff
		debug(LEVEL, "cost coeff:", self.cost_coeff)

		# Lets change it in case we need to take variable sign in account
		if self.variable_sign_needed:
			self.signed_variable = self.original_variable[:]
			debug(LEVEL, " self.signed_variable ", self.signed_variable)
			# -1 * coefficient for negative variables
			for var in self.negative_variables:
				var_index = self.signed_variable.index(var)
				self.cost_coeff[0, var_index] = -1 * self.cost_coeff[0, var_index]
				self.signed_variable[var_index] = var+'-'
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)
			# for free variables replace them by x+ - x-
			for var in self.free_variables:
				var_index = self.signed_variable.index(var)
				self.cost_coeff = np.insert(self.cost_coeff, var_index+1, -1 * self.cost_coeff[0, var_index], axis=1)
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)
				self.signed_variable[var_index] = var+'+'
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)
				self.signed_variable.insert(var_index+1, var+'-')
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)

			debug(LEVEL, "signed variable ", self.signed_variable)

	# get coefficient of variable from objective function or string
	def get_coeff(self, str_line, var):
		# get coefficient of var from str_line
		debug(LOCAL_LEVEL, "str_line ", str_line, " var ", var)
		# search for coefficient only if a variable is there in search str otherwise return 0
		search_str = re.search(var+'\\b', ''.join(str_line.split()))
		if(search_str):
			# get the coefficient of the form 123, +123, -123, -12.3, 2/3, -2/3 (.12 is invalid use 0.12 instead)
			# Regex! Oh Regex! what would I do without thou!
			coeff = re.findall('(\-?\d+\.*\d*|\-?\d+\/\.*\d*)('+var+'\\b)', ''.join(str_line.split()))
			# coeff = re.findall('(\-?\d+\.*\d*)('+var+'\\b)', ''.join(str_line.split()))
			# if there is no coefficient then its either +1 and -1 decide based on sign
			if len(coeff) == 0:
				sign = re.findall('(\-)('+var+'\\b)', ''.join(str_line.split()))
				if(len(sign) != 0 and sign[0][0] == '-'):
					coeff = [('-1', var)]
				else:
					coeff = [('1', var)]
			debug(LOCAL_LEVEL, "coeff of variable ", var, " in ", " constraint ", str_line, " is ", coeff)
			return float(Fraction(coeff[0][0]))
		else:
			return 0.

	# this function will get coefficient matrix and b from original problem, we won't worry about standardizing them yet
	def get_coeff_matrix(self):
		# get coefficient matrix A and also get RHS b
		self.coeff_matrix = np.zeros((len(self.constraints), len(self.original_variable)))
		self.b = np.zeros((len(self.constraints), 1))
		# b would be anything that is after <=, >= or =
		for i in range(len(self.constraints)):
			self.b[i, 0] = float(Fraction(re.split('<=|>=|=', self.constraints[i])[-1]))
		# generate coefficient matrix using get_coeff with constraints
		j_index = 0
		for var in self.original_variable:
			for i in range(len(self.constraints)):
				self.coeff_matrix[i, j_index] = self.get_coeff(self.constraints[i], var)
				debug(LOCAL_LEVEL, self.constraints[i], " ", var, " ", self.coeff_matrix[i, j_index])
			j_index = j_index+1
		debug(LEVEL, "Coefficient Matrix: \n", self.coeff_matrix, "\n b: \n", self.b)

		# Lets change it in case we need to take variable sign in account
		if self.variable_sign_needed:
			self.signed_variable = self.original_variable[:]
			debug(LEVEL, " self.signed_variable ", self.signed_variable)
			# as before -1* coefficients for negative variables
			for var in self.negative_variables:
				var_index = self.signed_variable.index(var)
				self.coeff_matrix[:, var_index] = -1 * self.coeff_matrix[:, var_index]
				self.signed_variable[var_index] = var+'-'
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)
			# replace free variables with x+ - x-, this essentially means adding extra column to coefficient matrix
			for var in self.free_variables:
				var_index = self.signed_variable.index(var)
				self.coeff_matrix = np.insert(self.coeff_matrix, var_index+1, -1 * self.coeff_matrix[:, var_index], axis=1)
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)
				self.signed_variable[var_index] = var+'+'
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)
				self.signed_variable.insert(var_index+1, var+'-')
				debug(LOCAL_LEVEL, " var_index ", var_index, " signed_variable: ", self.signed_variable, " var: ", var)

			debug(LEVEL, " self.coeff_matrix after sign adjustments \n", self.coeff_matrix)
			# now that we have taken variable signs in account, lets delete non_negativity constraints
			for i in self.constraints_for_sign:
				self.coeff_matrix = np.delete(self.coeff_matrix, i, 0)
				self.constraints.remove(self.constraints[i])
				self.b = np.delete(self.b, i, 0)
			debug(LEVEL, " self.coeff_matrix after sign adjustments and deleting additional rows\n", self.coeff_matrix)
		return self.constraints, self.coeff_matrix, self.b

	def get_standardized_coeff_matrix(self, constraints, coeff_matrix, b):
		# take the original coefficient matrix and
		# add slack variables to convert it to standard form
		# make sure there isn't any negative b
		is_two_phase_required = False
		coeff_matrix_with_slack = coeff_matrix
		b_positive = b
		# add positive slack with <= and negative slack with >= constraints
		# if all constraints are <= then we don't need Two Phase otherwise we do
		# if any of the b is negative we'll use Two Phase, in some cases we don't have to but
		# efforts to single out those cases are not worth the time
		for constraint in constraints:
			index = constraints.index(constraint)
			if '<=' in constraint:
				slack_column = np.eye(1, len(constraints), index).transpose()
				coeff_matrix_with_slack = np.append(coeff_matrix_with_slack, slack_column, 1)
				if b[index] < 0.:
					coeff_matrix_with_slack[index] = -1 * coeff_matrix_with_slack[index]
					b_positive[index] = -1 * b_positive[index]
					is_two_phase_required = True
			elif '>=' in constraint:
				slack_column = -1 * np.eye(1, len(constraints), index).transpose()
				coeff_matrix_with_slack = np.append(coeff_matrix_with_slack, slack_column, 1)
				if b[index] < 0:
					coeff_matrix_with_slack[index] = -1 * coeff_matrix_with_slack[index]
					b_positive[index] = -1 * b_positive[index]
				else:
					is_two_phase_required = True
			elif '=' in constraint:
				if b[index] < 0:
					coeff_matrix_with_slack[index] = -1 * coeff_matrix_with_slack[index]
					b_positive[index] = -1 * b_positive[index]
				is_two_phase_required = True
		coeff_matrix_with_slack += 0.
		debug(LEVEL, "standardized coeff matrix: \n", coeff_matrix_with_slack, "\nb_positive: \n", b_positive, "\nis_two_phase_required: ", is_two_phase_required)
		return is_two_phase_required, coeff_matrix_with_slack, b_positive

	# this is called if assume_non_negative_variable is False
	# gets the sign of variable
	def get_variable_sign(self):
		# tell others that they need to consider variable sign into the account
		self.variable_sign_needed = True
		# warn humans that they are asking you to do too much so just check their intention
		print '\033[91m' + "Make sure you know what you are doing." + '\033[0m'
		print "You chose assume_non_negative_variable = False."
		print "This means that any variable that is not explicitly mentioned in problem file will be trated as free variable."
		self.non_negative_variables = []
		self.negative_variables = []
		self.free_variables = []
		self.constraints_for_sign = []
		# we'll determine the sign based on the non-negativity (negativity) constraint
		# x>=0 or x<=0 will be matched and others will be assumed to be free
		# we'll also mark these constraints to remove them later
		for variable in self.original_variable:
			variable_sign = False
			for constraint in self.constraints:
				stripped_constraint = ''.join(constraint.split())
				debug(LOCAL_LEVEL, "constraint: ", stripped_constraint)
				positive_constraint = variable+'>=0'
				negative_constraint = variable+'<=0'
				if not variable_sign:
					if positive_constraint == stripped_constraint:
						self.non_negative_variables.append(variable)
						variable_sign = True
						debug(LOCAL_LEVEL, "positive ", variable, " ", stripped_constraint)
						self.constraints_for_sign.append(self.constraints.index(constraint))
					elif negative_constraint == stripped_constraint:
						self.negative_variables.append(variable)
						variable_sign = True
						debug(LOCAL_LEVEL, "negative ", variable, " ", stripped_constraint)
						self.constraints_for_sign.append(self.constraints.index(constraint))
			if not variable_sign:
				self.free_variables.append(variable)
				variable_sign = True
				debug(LOCAL_LEVEL, "free ", variable)
		print " positive variables: ", self.non_negative_variables
		print " negative variables: ", self.negative_variables
		print " free variables: ", self.free_variables
		self.constraints_for_sign.sort()
		self.constraints_for_sign.reverse()

if __name__ == '__main__':
	test_input = " minimize   3/4 x1 + 0.25 x2 - 0.4 x3\
	subject to x1 + x2 + 2x3 <= 9 \
	x1 + x2 - x3 <= 2\
	- x1 + x2 + x3 <= 4 \
	x1 >= 0\
	x2 <= 0 \
	"
	preprocessed_input = PreProcessing(test_input)
	preprocessed_input.parse_variables()
	preprocessed_input.get_variable_sign()
	preprocessed_input.get_original_cost_coeff()
	preprocessed_input.get_coeff_matrix()
	preprocessed_input.get_standardized_coeff_matrix(preprocessed_input.constraints, preprocessed_input.coeff_matrix, preprocessed_input.b)
