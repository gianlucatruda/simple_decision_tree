"""
Simple decision tree with scikit-learn,
using the Titanic survivorship data.
18 June 2018
"""
import requests
from sklearn import tree

# Importing the json data for Titanic survivors from the web
print("Retrieving Titanic survivorship data...")
try:

	r = requests.get("https://titanic.businessoptics.biz/survival")
	data = r.json()
	num_entries = len(data)
	print("Retrieved {} data points successfully.".format(num_entries))
	added = 0 # The number of entries successfully added
	excluded = 0 # The numeber of entries discarded

	# Creating two sets for our data
	X = [] # The input variables [passenger class, age, sex]
	Y = [] # The output variables [survived]

	# Adding complete data to our X and Y sets
	for passenger in data:
		try:
			p_class = int(passenger['class'])
			p_age = int(passenger['age'])
			p_is_male = 1 if passenger['sex'].lower() == 'male' else 0
			p_did_surv = passenger['survived']
			X.append([p_class, p_age, p_is_male])
			Y.append(p_did_surv)
			# print(X[-1], Y[-1])
			added += 1
		except:
			excluded += 1

	# Check data import successfulness
	print("Added: {}\tExcluded: {}\tMissing: {}".format(
		added, excluded, num_entries - added - excluded
		))

	# instantiate classifier
	classifier = tree.DecisionTreeClassifier(max_depth=5)

	# fit X to Y
	classifier = classifier.fit(X, Y)

	# predict Y based on unseen X values
	print("\nMaking some predictions...")
	test_cases = [
		[1, 55, 0], # 55 year old woman in first class
		[2, 48, 1], # 48 year old man in second class
		[3, 20, 1], # 20 year old man in third class
	]
	prediction = classifier.predict(test_cases)
	for key, p in enumerate(prediction):
		print(test_cases[key], p)

	# Create a .dot file of the decision tree
	tree.export_graphviz(classifier, out_file="myDT.dot")

	"""
	To convert the .dot file to a .png, install graphviz to machine
	and run something along the lines of:
	$ dot -Tpng myDT.dot -o myDT.png
	"""

except:
	print("Error: Please ensure internet connection and retry.")
