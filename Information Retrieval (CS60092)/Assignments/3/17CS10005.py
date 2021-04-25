def getMutualInformation(tok):
	MI = 0

	classname = ""
	otherclass = ""
	if tok in Token_list["class1"]["train"]:
		classname="class1"
		otherclass="class2"
	else:
		classname="class2"
		otherclass="class1"

	docCount = N_document["class1"]["train"] + N_document["class2"]["train"]

	hasterm_class_same = 0
	nothasterm_class_same = N_document[classname]["train"]
	if tok in DF_token[classname]["train"]:
		hasterm_class_same = DF_token[classname]["train"][tok]
		nothasterm_class_same -= DF_token[classname]["train"][tok]
	
	hasterm_class_diff = 0
	nothasterm_class_diff = N_document[otherclass]["train"]
	if tok in DF_token[otherclass]["train"]:
		hasterm_class_diff = DF_token[otherclass]["train"][tok]
		nothasterm_class_diff -= DF_token[otherclass]["train"][tok]

	N = [[0,0],[0,0],[0,0],[0,0]]
	N[0][0] = nothasterm_class_diff
	N[0][1] = nothasterm_class_diff
	N[1][0] = hasterm_class_diff
	N[1][1] = hasterm_class_diff

	n = [0, 0]
	n[0] = N[0][0] + N[0][1] + 1
	n[1] = N[1][0] + N[1][1] + 1

	c = [0, 0]
	c[0] = N[0][0] + N[1][0] + 1 
	c[1] = N[0][1] + N[1][1] + 1

	#Added 1's to handle the math errors
	#Won't affect the ans

	for i in [0, 1]:
		for j in [0, 1]:
			MI += (N[i][j]/docCount) * math.log(docCount*N[i][j] + 1) / (n[i] * c[j])

	return MI


def featureSelection(V, k):
	term_MI = []

	for tok in V:
		mi = getMutualInformation(tok)
		term_MI.append([tok, mi])

	term_MI = sorted(term_MI, key = lambda x: x[1], reverse = True)[:2*k]

	feature = set()
	for item in term_MI:
		feature.add(item[0])

	return feature


def trainMultinomailNB(k):
	prior = dict()
	condprob = dict()
	V = set()

	V = Token_list["class1"]["train"]
	for tok in Token_list["class2"]["train"]:
		V.add(tok)

	N = N_document["class1"]["train"] + N_document["class2"]["test"]
	N_c_1 = N_document["class1"]["train"]
	prior["class1"] = N_c_1/N
	N_c_2 = N_document["class2"]["train"]
	prior["class2"] = N_c_2/N

	denominator_sum = 0
	for tok in V:
		T_ct_1 = 0
		if tok in Token_count["class1"]["train"].keys():
			T_ct_1 = Token_count["class1"]["train"][tok]

		T_ct_2 = 0
		if tok in Token_count["class2"]["train"].keys():
			T_ct_2 = Token_count["class2"]["train"][tok]

		denominator_sum += T_ct_2 + T_ct_1 + 1

	condprob["class1"] = dict()
	for tok in V:
		if tok in Token_count["class1"]["train"].keys():
			T_ct_1 = Token_count["class1"]["train"][tok] + 1
			condprob["class1"][tok] = T_ct_1 / denominator_sum

	condprob["class2"] = dict()
	for tok in V:
		if tok in Token_count["class2"]["train"].keys():
			T_ct_2 = Token_count["class2"]["train"][tok] + 1
			condprob["class2"][tok] = T_ct_2 / denominator_sum

	features = featureSelection(V, k)
	return V, prior, condprob, features


def applyMultinomialNB(V, prior, condprob, features):
	expected_out = []
	naive_out = []

	for doc in Doc_term_tf_idf["class1"]["test"].keys():
		expected_out.append(1)

		score_c1 = math.log(prior["class1"])
		score_c2 = math.log(prior["class2"])
		for tok in Doc_term_tf_idf["class1"]["test"][doc].keys():
			if tok in features:
				if tok in condprob["class1"].keys():
					score_c1 += math.log(condprob["class1"][tok])
				if tok in condprob["class2"].keys():
					score_c2 += math.log(condprob["class2"][tok])

		if score_c1 < score_c2:
			naive_out.append(1)
		else:
			naive_out.append(2)


	for doc in Doc_term_tf_idf["class2"]["test"].keys():
		expected_out.append(2)

		score_c1 = math.log(prior["class1"])
		score_c2 = math.log(prior["class2"])
		for tok in Doc_term_tf_idf["class2"]["test"][doc].keys():
			if tok in features:
				if tok in condprob["class1"].keys():
					score_c1 += math.log(condprob["class1"][tok])
				if tok in condprob["class2"].keys():
					score_c2 += math.log(condprob["class2"][tok])

		if score_c1 < score_c2:
			naive_out.append(1)
		else:
			naive_out.append(2)

	accuracy = accuracy_score(naive_out, expected_out) * 100
	recall = recall_score(naive_out, expected_out) * 100
	print(f"The accuracy for Naive MultinomialNB {accuracy} and recall {recall}")

	return accuracy, recall


def trainBernoulliNB(k):
	prior = dict()
	condprob = dict()
	V = set()

	V = Token_list["class1"]["train"]
	for tok in Token_list["class2"]["train"]:
		V.add(tok)

	N = N_document["class1"]["train"] + N_document["class2"]["test"]
	N_c_1 = N_document["class1"]["train"]
	prior["class1"] = N_c_1/N
	N_c_2 = N_document["class2"]["train"]
	prior["class2"] = N_c_2/N

	condprob["class1"] = dict()
	for tok in V:
		if tok in Token_list["class1"]["train"]:
			T_ct_1 = 1
			if tok in DF_token["class1"]["train"].keys():
				T_ct_1 += DF_token["class1"]["train"][tok]
			condprob["class1"][tok] = T_ct_1 / (N_c_1 + 2)

	condprob["class2"] = dict()
	for tok in V:
		if tok in Token_list["class2"]["train"]:
			T_ct_2 = 1
			if tok in DF_token["class2"]["train"].keys():
				T_ct_2 += DF_token["class2"]["train"][tok]
			condprob["class2"][tok] = T_ct_2 / (N_c_2 + 2)

	features = featureSelection(V, k)
	return V, prior, condprob, features 


def applyBernoulliNB(V, prior, condprob, features):
	expected_out = []
	naive_out = []

	for doc in Doc_term_tf_idf["class1"]["test"].keys():
		expected_out.append(1)

		score_c1 = math.log(prior["class1"])
		score_c2 = math.log(prior["class2"])
		for tok in features:
			if tok in Doc_term_tf_idf["class1"]["test"][doc].keys():
				if tok in condprob["class1"].keys():
					score_c1 += math.log(condprob["class1"][tok])
				if tok in condprob["class2"].keys():
					score_c2 += math.log(condprob["class2"][tok])
			else:
				if tok in condprob["class1"].keys():
					score_c1 += math.log(1 - condprob["class1"][tok])
				if tok in condprob["class2"].keys():
					score_c2 += math.log(1 - condprob["class2"][tok])

		if score_c1 > score_c2:
			naive_out.append(1)
		else:
			naive_out.append(2)


	for doc in Doc_term_tf_idf["class2"]["test"].keys():
		expected_out.append(2)

		score_c1 = math.log(prior["class1"])
		score_c2 = math.log(prior["class2"])
		for tok in features:
			if tok in Doc_term_tf_idf["class2"]["test"][doc].keys():
				if tok in condprob["class1"].keys():
					score_c1 += math.log(condprob["class1"][tok])
				if tok in condprob["class2"].keys():
					score_c2 += math.log(condprob["class2"][tok])
			else:
				if tok in condprob["class1"].keys():
					score_c1 += math.log(1 - condprob["class1"][tok])
				if tok in condprob["class2"].keys():
					score_c2 += math.log(1 - condprob["class2"][tok])

		if score_c1 > score_c2:
			naive_out.append(1)
		else:
			naive_out.append(2)

	accuracy = accuracy_score(naive_out, expected_out) * 100
	recall = recall_score(naive_out, expected_out) * 100
	print(f"The accuracy for Naive BernoulliNB {accuracy} and recall {recall}")

	return accuracy, recall


#------------------------------------#
#----------Main Code Starts----------#
#------------------------------------#
# curr_dir = os.getcwd()
# loc = os.path.join(curr_dir, 'dataset')
# /home/osboxes/Downloads/Assign_3/dataset
# For actual code
loc = sys.argv[1]

#______
# Ye Tera Assignement 2 se preprocess karna h 
# Doc_term_tf_idf ---> doc_id & term to tf_idf score per class & data type
# Token_list ---> a dict of all token per class & data type, set()
# IDF_score --->  a dict of all token per class & data type and their IDF scores
# Token_count ---> a dict of all token per class & data type and their number of apperances in this class
# N_token number of token in a class
# N_document number of document in a class
# DF_token ---> tok to number of doc it appear in 

Doc_term_tf_idf = dict()
Token_list = dict()
IDF_score = dict()
Token_count = dict()
DF_token = dict()
N_token = dict()
N_document = dict()

#________
#Yaha tak



# ---------Ye part 1 ka ans h------

preProcessData("class1", "train")
preProcessData("class2", "train")
preProcessData("class1", "test")
preProcessData("class2", "test")

# print(IDF_score["class2"]["test"])
# print("\n\n\n\n")

# print(Token_list["class2"]["train"])
# print("\n\n\n\n")

# print(Doc_term_tf_idf["class1"]["train"])
# print("\n\n\n\n")


print("Creating Training/Testing Data...")

filename = sys.argv[2]
output_file = open(filename, 'w')
K = [1, 10, 100, 1000, 10000]

for k in K:
	print(f"Processing for K = {k} :: ")
	output_file.write(f"For Number of feature K = {k}\n")
	output_file.write("MultinomialNB :")

	V, prior, condprob, features = trainMultinomailNB(k)
	accuracy, recall = applyMultinomialNB(V, prior, condprob, features)

	F1 = 2 * (accuracy * recall) / (accuracy + recall + 1)
	output_file.write(f"F1 Score = {F1}  Accuracy = {accuracy}  Recall = {recall}")
	output_file.write('\n\n')


	output_file.write("BernoulliNB :")

	V, prior, condprob, features = trainBernoulliNB(k)
	accuracy, recall = applyBernoulliNB(V, prior, condprob, features)

	F1 = 2 * (accuracy * recall) / (accuracy + recall + 1)
	output_file.write(f"F1 Score = {F1}  Accuracy = {accuracy}  Recall = {recall}")
	output_file.write('\n\n')
	print("\n")

output_file.close()
