from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

tree = DecisionTreeClassifier(criterion = 'gini',min_samples_leaf=10, min_samples_split=20)
adaboostcla = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=250,learning_rate=1e-1)
xgboostcla = XGBClassifier(learning_rate=1e-1,n_estimators=500,nthread=2,max_depth=3,reg_alpha=1e-4,gamma=1e-1)
graboostcla = GradientBoostingClassifier(n_estimators=250, max_depth=5,min_samples_leaf=2, learning_rate=1e-1)
clf = [tree,adaboostcla,xgboostcla,graboostcla]
clf_name = ['Decision Tree','Adaboost','Xgboost','Gboost']

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion = 'gini',n_estimators = 1250,max_depth=12,min_samples_leaf=2, min_samples_split=2)