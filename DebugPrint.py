# Author : Adarsh (abarik@purdue.edu)
# prints debug statements to help debug the code


def debug(level=0, *args):
	if level == 1:
		print "".join(map(str, args))
	elif level == 2:
		# prints in Red
		print '\033[91m' + "".join(map(str, args)) + '\033[0m'


if __name__ == '__main__':
	a = "hahaha"
	b = 5.3
	debug(0, "this won't get print!", a, b)
	debug(1, "this is debug print!", a, b)
	debug(2, "this is error print!", a, b)
