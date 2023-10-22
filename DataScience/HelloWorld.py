#import seaborn as sb
#import numpy as np
#from matplotlib import pyplot as plt

#df = sb.load_dataset('iris')
#sb.jointplot(x='petal_length', y='petal_width', data=df, kind='kde')
# plt.show()
# def sinplot(flip=1):
#x = np.linspace(0, 14, 100)
# for i in range(1, 5):
#plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

#sb.set_style("darkgrid", {'axes.axisbelow': False})
# sinplot()
# sb.despine()
#import seaborn as sb
#from matplotlib import pyplot as plt
#df = sb.load_dataset('iris')
# print(df)
# sb.set_style("ticks")
#sb.pairplot(df, hue='species', diag_kind="kde", kind="scatter", palette="husl")
# plt.show()
#from sklearn.datasets import load_iris
#iris = load_iris()
#X = iris.data
#y = iris.target
#feature_names = iris.feature_names
#target_names = iris.target_names
#print("Feature names:", feature_names)
#print("Target names:", target_names)
#print("\nFirst 10 rows of X:\n", X[:10])
#import matplotlib.pyplot as plt
#import numpy as np
# np.random.seed(1)

#x = np.random.rand(15)
#y = np.random.rand(15)
#names = np.array(list("ABCDEFGHIJKLMNO"))
#c = np.random.randint(1, 5, size=15)

#norm = plt.Normalize(1, 25)
#cmap = plt.cm.RdYlGn

#fig, ax = plt.subplots()
#sc = plt.scatter(x, y, c=c, s=100, cmap=cmap, norm=norm)

# annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
#bbox=dict(boxstyle="round", fc="w"),
# arrowprops=dict(arrowstyle="->"))
# annot.set_visible(False)


# def update_annot(ind):

#   pos = sc.get_offsets()[ind["ind"][0]]
#    annot.xy = pos
#   text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
#                         " ".join([names[n] for n in ind["ind"]]))
# annot.set_text(text)
# annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
# annot.get_bbox_patch().set_alpha(0.4)


# def hover(event):
#   vis = annot.get_visible()
#  if event.inaxes == ax:
#     cont, ind = sc.contains(event)
#    if cont:
# annot.set_visible(True)
#     fig.canvas.draw_idle()
# else:
#   if vis:
#      annot.set_visible(False)
#     fig.canvas.draw_idle()


#fig.canvas.mpl_connect("motion_notify_event", hover)

# plt.show()

#df = pd.read_csv('pokemon_data.csv')
#dg = df.head(15)
#sb.pointplot(x="Attack", y="Defense", data=dg)
# plt.show()

import opendatasets

opendatasets.download('https://www.kaggle.com/c/rossmann-store-sales')
