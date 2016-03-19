# Python implementation of two phase Simplex algorithm to solve linear programs

This repository contains python implementation of two phase Simplex algorithm. The cool thing about this algorithm is that it can take input as a mathematical model i.e. we can write variables and constraints like a mathematical mdoel and it can make sense out of it. Ofcourse, it also support data input as matrices and vector in standard form. 

For example check this input to solve [this problem](http://www.math.washington.edu/~burke/crs/407/models/m1.html)

```
# comments start with # no inline comments
# numbers can be integer (1, 2, -3), fractions (1/2, -2/4, 3/5) or decimals (1.2, 2.3, 0.32)
# note decimal as .123 are not allowed so don't be lazy and write them as 0.123

# objective function, you can use min, minimize, min. or max, maximize, max. (case insensitive)
# Lets say that xij is number of product i manufactured at center j
# i = 1 (large), 2(medium), 3(small)
# j = 1 (center 1), 2 (center 2), 3 (center 3)

maximize 12 x11 + 12 x12 + 12 x13 + 10 x21 + 10 x22 + 10 x23 + 9 x31 + 9 x32 + 9 x33   

# subject to, such that, sub to (case insensitive)

subject to 

# fraction (scheduled production)/(center's capacity) should be same at all the centers
1/550 x11 + 1/550 x21 + 1/550 x31 - 1/750 x12 - 1/750 x22 - 1/750 x32 = 0
1/750 x12 + 1/750 x22 + 1/750 x32 - 1/275 x13 - 1/275 x23 - 1/275 x33 = 0

# centers should not exceed their production capacities
x11 + x21 + x31 <= 550
x12 + x22 + x32 <= 750
x13 + x23 + x33 <= 275

# there is no point producing a product more than its market demand
x11 + x12 + x13 <= 700
x21 + x22 + x23 <= 900
x31 + x32 + x33 <= 340

# each center also has constraint on water availability
21 x11 + 17 x21 + 9 x31 <= 10000
21 x12 + 17 x22 + 9 x32 <= 7000 
21 x13 + 17 x23 + 9 x33 <= 4200

# non-negativity constraints (can be skipped if assume_non_negative_variables is set to True)
# write 0 not 0. or 0.0000, don't say later that I didn't tell you beforehand

# variables will be obviously non-negative, we will take care of this from config
```

## Required configuration
Configurations are read from a simple `config` file (Check this file in repository for more details). Main configuration is to let the program know about the input method 

```
use_human_readable_config = True
use_standard_form_config = False
```

## Code
* `SimplexTwoPhase.py` : Main program which needs to run to solve the problem
* `SimplexMain.py` : Main Simplex routine
* `Preprocessing.py` : To process the input (Regex to make sense of input)
* `DebugPrint.py` : Some helper function

## How to run it
* Make sure all python codes and `config` file are in a single directory – lets call that `test directory`.
* Open terminal (command line) in Linux machine
* Change directory to test directory
`$ cd (path to )‘test directory’`
* Check `config` file and make sure that all configuration are correct (for example check that file paths are correct).
* Once you are satisfied with all configuration, run `SimplexTwoPhase.py` program.
`$ python SimplexTwoPhase.py`
* Result will be displayed in the standard output.

## Output
A sample output will look like this (for the problem mentioned above):
```
OPTIMAL VALUE	12823.6470588
x11 	419.294117647
x22 	231.764705882
Variable  10 	130.705882353 	<-- Slack Variable
Variable  11 	178.235294118 	<-- Slack Variable
Variable  12 	65.3529411765 	<-- Slack Variable
Variable  13 	121.705882353 	<-- Slack Variable
Variable  14 	617.588235294 	<-- Slack Variable
x23 	50.6470588235
Variable  16 	1194.82352941 	<-- Slack Variable
x13 	159.0
x32 	340.0
```