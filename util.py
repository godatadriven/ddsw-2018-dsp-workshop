import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
from scipy import ndimage
from sklearn import *
from matplotlib import pylab

from IPython.core.display import display, HTML
display(HTML("<style>.container {width:80% !important; }</style>"))



def xlabel(l, fontsize='x-large'):
    pylab.xlabel(l, fontsize=fontsize)


def ylabel(l, fontsize='x-large'):
    pylab.ylabel(l, fontsize=fontsize)


def legend(l, fontsize='x-large'):
    pylab.legend(l, fontsize=fontsize, loc='best')


def title(l, fontsize='x-large'):
    pylab.title(l, fontsize=fontsize)


def figsize(c=16, r=3):
    pylab.rcParams['figure.figsize'] = c,r


def no_xticks():
    pylab.xticks([])


def no_yticks():
    pylab.yticks([])


def no_ticks():
    no_xticks(); no_yticks();


def imshow(im):
    pylab.imshow(nxmx3(im))
    no_ticks()


def imread(filename, mode='RGB', edit_dir=True):
    if edit_dir:
        if not filename.startswith('images/'):
            filename = 'images/' + filename
    # just do it, introduce more care later
    return ndimage.imread(filename, mode=mode).astype('float')/255.


def image_subplots(images, titles=None):
    figsize(20, 6)
    for i in range(len(images)):
        pylab.subplot(1, len(images), i+1);
        imshow(images[i]);
        if titles is not None:
            title(titles[i]);
    figsize()


def float3d_to_uint2d(im):
    return pylab.array(pylab.mean(im, 2)*255, dtype='uint8')


# make sure the given matrix is MxNx3
def nxmx3(m):
    sh = m.shape
    m2 = m.copy() # working copy, might chance
    assert(len(sh) == 2 or len(sh) == 3)
    if len(sh) == 3:
        if sh[2] == 3:
            # matrix is already mxnx3
            return m2
        else:
            # matrix is MxNxD with D!=3, best we can do:
            m2 = pylab.mean(m2, 2)
    # here, m2 is MxN...
    return pylab.tile(m2.reshape(sh[0], sh[1], 1), (1, 1, 3))


# scale the image values, not the dimensions
def imscale(im):
    mi = min(im.flatten())
    return (im - mi) / (max(im.flatten()) - mi)


# detect the edges in an image comprised of floating point values
# derivative in {'x','y','grad'}
def image_derivative(im, derivative, do_scale=True):
    assert(len(im.shape)==2 or len(im.shape)==3)
    assert(im.dtype=='float')
    assert(derivative in ['x', 'y', 'grad'])
    # obtain a gray image
    if len(im.shape) == 3:
        gray_im = pylab.mean(im, 2)
    else:
        gray_im = im
    # either x,y derivative or full gradient
    if derivative == 'grad':
        dy = ndimage.sobel(gray_im, 0)
        dx = ndimage.sobel(gray_im, 1)
        deriv_im = pylab.sqrt(dy*dy+dx*dx)
    else:
        deriv_im = ndimage.sobel(gray_im, {'x': 1, 'y': 0}[derivative])
    return imscale(deriv_im) if do_scale else deriv_im


# detect_edges either returns a binary mask with edges,
# or overlays the edges on the image
def detect_edges(im, detection_threshold=0.6, do_dilation=False, overlay=True):
    edges = (pylab.absolute(image_derivative(im, 'x', False)) > detection_threshold) | \
            (pylab.absolute(image_derivative(im, 'y', False)) > detection_threshold)
    if do_dilation:
        edges = ndimage.binary_dilation(edges)
    if overlay:
        # overlay edges in a color
        im2 = nxmx3(im)
        im2[edges] = [1, .2, .2]
        edges = im2
    return edges


# settings for 1D signal stuff
data_dir = 'data'
data_filenames = ['chickenpox_newyork.csv', 'measles_baltimore.csv', 'mumps_newyork.csv']
data_labels = {i:s[:-4] for i,s in zip(range(len(data_filenames)), data_filenames)}
data_variables = ['year', 'month', 'count']

# functionality for obtaining a random number within a certain range
class DatasetParam:
    def __init__(self, min_val, max_val=None):
        self.min_val = float(min_val)
        self.max_val = self.min_val if max_val is None else float(max_val)
        assert(self.min_val <= self.max_val)

    def __str__(self):
        return 'DataParam valued {:.2f} upto {:.2f}'.format(self.min_val, self.max_val)

    def val(self, as_int=True):
        v = self.min_val + pylab.random_sample(1) * (self.max_val - self.min_val)
        return v if not as_int else int(v)


# get the (lowest) index from the row in DataFrame df with the given values
def get_index(df, values):
    idx_arr = pd.Series(True, index=df.index)
    for k in values:
        idx_arr = idx_arr & (df[k]==values[k])
    idx = df[idx_arr].index
    return None if len(idx)==0 else idx[0]


# test it...
test_df = pd.DataFrame(pylab.array([[2000, 1], [2001, 1], [2001, 12], [2002, 2]]), columns=['a','b'])
assert(get_index(test_df, {'a': 2001, 'b': 12}) == 2)


# split the DataFrame @df containing a datamarket health series into many shorter signals
# splitting starts from yyyy-mm @sy-@sm and ends at @ey-@em
# the resulting signals are of length @size, and sampling occurs every @stride steps
# only the signal itself is maintained - the actual date could be informative but that does not matter in this lecture
# all 'numeric' parameters are of type DatasetParam
def split_datamarket_health_series(df, syear, smonth, eyear, emonth, stride, size):

    # check validity of parameters
    assert(isinstance(df, pd.DataFrame) and all(df.columns == data_variables))
    for param in [syear, smonth, eyear, emonth, stride, size]:
        assert(isinstance(param, DatasetParam))
    assert((syear.val()    <= eyear.val()) & \
           (smonth.val()   <= emonth.val()) & \
           (stride.min_val >  0) & \
           (size.min_val   >  0))

    # make sure we can start and end...
    start_idx = get_index(df, {'year': syear.val(), 'month': smonth.val()})
    end_idx = get_index(df, {'year': eyear.val(), 'month': emonth.val()})
    assert(start_idx is not None and end_idx is not None and start_idx <= end_idx)

    # ... and that all of this makes sense wrt the dataframe's index
    # i.e. the index consists of stepwise increasing integers
    df.index = range(df.shape[0])

    # loop over the signal to cut it into pieces
    signals = []
    while True:
        next_idx = start_idx + size.val()
        if next_idx >= df.shape[0]:
            break
        signals.append(df.iloc[start_idx : next_idx]['count'].values)
        start_idx += stride.val()

    return signals


# create a dataset from the datamarket health series
# samples consist of a label, associated to a variable-length series of numeric measurements
def make_datamarket_health_dataset(stride=DatasetParam(1, 4), size=DatasetParam(24, 36)):
    signals = []
    labels = []
    for filename, label in zip(data_filenames, data_labels.keys()):
        df = pd.read_csv(os.path.join(data_dir, filename), names=data_variables)
        signals2 = split_datamarket_health_series(df, DatasetParam(min(df.year)), DatasetParam(1),
                                                      DatasetParam(max(df.year)), DatasetParam(1),
                                                      stride, size)
        # store
        signals += signals2
        labels += [label for i in range(len(signals2))]
    return pylab.array(signals), pylab.array(labels)


# using leave-1-out model validation with your favorite classifier and normalizer
def l1o_model_validation(data,
                         labels,
                         classifier = neighbors.KNeighborsClassifier(1),
                         normalizer = preprocessing.StandardScaler()):
    l1o = cross_validation.LeaveOneOut(data.shape[0])
    predictions = pylab.zeros(labels.shape, dtype=labels.dtype)
    for train_idx, test_idx in l1o:
        classifier.fit(normalizer.fit_transform(data[train_idx]), labels[train_idx])
        predictions[test_idx] = classifier.predict(normalizer.transform(data[test_idx]))
    return sum(predictions == labels) / float(data.shape[0])
