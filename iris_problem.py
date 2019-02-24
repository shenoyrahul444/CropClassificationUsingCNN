from  sklearn import  datasets
iris=datasets.load_iris()
print("Data",iris.data)
print("Target",iris.target)
print("Target Names",iris.target_names)