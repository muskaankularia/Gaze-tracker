# this is linear kernel

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from matplotlib.colors import Normalize
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import pandas as pd

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#
# dataset for grid search
dataset= pd.read_csv('datax.csv')     #first row should be label (or should be left empty
#temptestdata= pd.read_csv('tempTest.csv')   #first row should be label (or should be left empty


Xtotal=dataset.iloc[:,:-1].values          #using all features
#Xtotal=dataset.iloc[:,:10].values       # using first 10 features
print Xtotal[0,:],Xtotal[-1,:],len(Xtotal)  #print 1st value of Xtotal, its last val and its length
ytotal=dataset.iloc[:,-1].values
print ytotal[0], ytotal[-1]
#Xtemptestdata=temptestdata
#------- to shuffle and divide data into train and test
# (train data is then further to be divide into train and cross val data)
from sklearn.model_selection import train_test_split
X, Xtestingdata, y, ytestingdata = train_test_split(Xtotal, ytotal, test_size=0.2, random_state=5)
#---------feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
Xtestingdata=scaler.transform(Xtestingdata)
#Xtemptestdata=scaler.transform(Xtemptestdata)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# Xtestingdata = scaler.fit_transform(Xtestingdata)
# #############################################################################
# Train classifiers
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

# C_range = np.logspace(-2, 5, 8)
# gamma_range = np.logspace(-7, 3, 11)

C_range = 10. ** np.arange(-4, 4)
E_range = 10. ** np.arange(-4, 4)
# C_range = 2. ** np.arange(1, 7)
# gamma_range = 2. ** np.arange(-7, 1)
# C_range=[]
# for i in range(-2,2):
#     C_range.append(10**i)
#     C_range.append(3*(10**i))
# gamma_range = C_range
print C_range
param_grid = dict(epsilon=E_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVR(kernel='linear'), param_grid=param_grid, cv=cv)
grid.fit(X, y)

required_params=grid.best_params_
print("The best parameters are %s with a score of %0.6f"
      % (required_params, grid.best_score_))
print 'best C is : %s n best epsilon is : %s' %(required_params['C'],required_params['epsilon'])
clf = SVR(C=required_params['C'], epsilon=required_params['epsilon'])
#clf = SVC(C=10, gamma=0.001)
clf.fit(X, y)
testing_acc=clf.score(Xtestingdata, ytestingdata)
print 'testing acc: %0.6f' % (testing_acc)
#predicted_classes= clf.predict(Xtemptestdata)
'''
###---------writing output to csv
idvals=np.arange(0,20)
for ival in idvals:
    row = [str(idvals[i]) + ',' + str(predicted_classes[i]) + '\n' for i in range(len(idvals))]
#print row
download_dir = "exampleCsv11.csv"  # where you want the file to be downloaded to
csv = open(download_dir, "w")
# "w" indicates that you're writing strings to the file
columnTitleRow = "id,class\n"
csv.write(columnTitleRow)
for r in row:
    csv.write(r)
#-----end writing to csv
'''



#print grid.cv_results_['mean_test_score']
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(E_range))

# # Draw heatmap of the validation accuracy as a function of gamma and C
# #
# # The score are encoded as colors with the hot colormap which varies from dark
# # red to bright yellow. As the most interesting scores are all located in the
# # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# # as to make it easier to visualize the small variations of score values in the
# # interesting range while not brutally collapsing all the low score values to
# # the same color.
#
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
#            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
# plt.xlabel('gamma')
# plt.ylabel('C')
# plt.colorbar()
# plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
# plt.yticks(np.arange(len(C_range)), C_range)
# plt.title('Validation accuracy')
# plt.show()
