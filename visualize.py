from __future__ import print_function
import numpy as np
from matplotlib import pyplot


def extract_acy(line):
	acy = line.strip('\n').split(' ')[-1]
	acy = float(acy[1:-2])
	return acy

def visualize(filename, color):
	start = False
	accuracies = []
	count = 0
	with open(filename) as infile:
		for line in infile:
			if "Accuracy of D" in line:
				start = True
			elif start and line.startswith("Test set:"):
				count += 1
				if count == 1:
					print(filename + " " + color + 
							"\t\tTeacher: {:.2f} \t\tStudent: ".format(extract_acy(line)), 
							end=' ')
				elif count%2 == 0:
					accuracies.append( extract_acy(line) )

	accuracies = np.array(accuracies)
	print(accuracies.max())
	pyplot.plot(accuracies, color)


def visualize_multi(filename, color):
	accuracies = []
	has_teacher = False
	with open(filename) as infile:
		for line in infile:
			# if "Teacher Test Accuracy:" in line and not has_teacher:
			# 	print(filename + " " + color + 
			# 			"\t\tTeacher: {:.2f} \t\tStudent: ".format())
			if line.startswith("Accuracy of Merged Students:"):
				accuracies.append( extract_acy(line) )

	accuracies = np.array(accuracies)
	print(accuracies.max())
	pyplot.plot(accuracies, color)


#visualize('/Users/chen/Desktop/parallel_features_deeperD_shuffledata.txt', 'red')
visualize('/Users/chen/Documents/Dropbox alias/AdversarialCompression/parallel_deeperD_shuffledata.txt', 'blue')
visualize('/Users/chen/Desktop/parallel_featuresconv1_deeperD_shuffledata.txt', 'green') #could be good
visualize('/Users/chen/Desktop/parallel_Dcnnfeatures_deeperD_shuffledata.txt', 'black') #could be good
#visualize_multi('/Users/chen/Desktop/batch_3students_deeperD2.txt', 'pink')
visualize('/Users/chen/Desktop/parallel_Dcnnfeatures_lr0.05_deeperD_shuffledata.txt', 'red')
visualize('/Users/chen/Desktop/parallel_Dcnnfeatures_lr0.0025_deeperD_shuffledata.txt', 'pink')
visualize('/Users/chen/Desktop/parallel_deeperD_shuffledata2.txt', 'orange')
#visualize('/Users/chen/Desktop/parallel_deeperD2_shuffledata.txt', 'pink')
pyplot.show()