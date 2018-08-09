import process as p
from sklearn import datasets, svm, metrics, utils
from sklearn.ensemble import RandomForestClassifier


dataset = p.load_data("./pot.csv","./targets.csv")

print("hello there :)")

#clf = svm.SVC()
clf = RandomForestClassifier(max_depth=5, random_state=0)

print("100% of the data is {}.".format(len(dataset.data)))

# Get 4/5
split_index = len(dataset.data)//5*4
print("80% of the data is {}.".format(split_index))

train_data = dataset.data[:split_index]
test_data = dataset.data[split_index:]

train_target = dataset.target[:split_index]
test_target = dataset.target[split_index:]

print train_target.shape
clf.fit(train_data, train_target)

out = clf.predict(test_data[0:])

print out

print test_target