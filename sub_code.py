#connecting to the code file

programfile = open('/root/mymodel/model_code.py',r)
code = programfile.read()				

if 'keras' or 'tensorflow' in code:			
	if 'Convolutional2D' in code:				 
		print('CNN')
	else:
		print('not CNN')
else:
	print('not deep learning')
