#####################################################################################################################################################
#																		    #
#  This script reads in the modules used by an ipython notebook (given as argument at the cmd line) and produces the following inline output info   #
#																		    #
#		1. Author, 2. Date, 3. Modules used and corresponding version info, 4. Python and IPython versions, 5. System info                  #
#																		    #
#####################################################################################################################################################


# Imports
import sys
from pkg_resources import get_distribution
import datetime
import IPython
import platform
from multiprocessing import cpu_count


# Define author
author = 'Peter Winslow'

# Define timestamp
now = datetime.datetime.now()
time_stamp = now.strftime('%A %B %d %Y')

# Read in list of modules from cmd line
mod_list =  sys.argv[1:]

# Get version info for each module in mod_list
mod_info = []
for mod in mod_list:
	mod_info.append(get_distribution(mod))

# Print all information

print '\nGeneral Information...'
print 'Author: {0}'.format(author)
print 'Date: {0}'.format(time_stamp)

print '\nPython Information...'
print 'CPython: {0}'.format(platform.python_version())
print 'IPython: {0}'.format(IPython.__version__)

print '\nModule Information...'
for mod in range(len(mod_info)):
	print mod_info[mod]

print '\nSystem Information...'
print 'Compiler: {0}'.format(platform.python_compiler())
print 'System: {0}'.format(platform.system())
print 'Release: {0}'.format(platform.release())
print 'Machine: {0}'.format(platform.machine())
print 'Processor: {0}'.format(platform.processor())
print 'CPU Cores: {0}'.format(cpu_count())
print 'Interpreter: {0}'.format(platform.architecture()[0])
