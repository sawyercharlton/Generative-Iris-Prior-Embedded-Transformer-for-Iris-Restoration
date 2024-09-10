#-*-coding:utf-8-*--
import os
import shutil

source_dir = '' 
destination_dir = ''
key = '61978'

def gather(source_dir):
	for root, dirs, files in os.walk(source_dir):
		for dir in dirs:
			gather(root + dir)
		for file in files:
			if key in file:
				old_name = root + os.path.sep + file
				new_name = destination_dir + os.path.sep + file
				shutil.copyfile(old_name, new_name)

gather(source_dir)
 