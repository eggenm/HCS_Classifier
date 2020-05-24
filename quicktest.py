import pandas as pd
import itertools as it
import numpy as np
import glob
import rioxarray as rx
import data_helper as dh
import timer
import scipy.stats as stat


def return_window(img, i, j, n):
    """
    Parameters
    ----------
    array: np array
        Array of image to pull from

    i: int
        row location of center

    j: int
        column location of center

    n: int
        width of moving window

    Returns
    -------
    window: np array
        nxn array of values centered around pixel i,j
    """
    shift = (n - 1) / 2
    window = img[:, int(i - shift):int(i + shift + 1), int(j - shift):int(j + shift + 1)]
    return window




def gen_windows2(array):
    x=range(0, array.shape[1])
    y=range(0, array.shape[2])
    tuples = list(it.product(x, y))
    myfunc = lambda a: a.flatten()
    aList = [myfunc(array[i, :, :]) for i in range(0, array.shape[0])]
    full_index = pd.MultiIndex.from_product([x, y], names=['i', 'j'])
    i = 0
    windows = pd.DataFrame({i: aList[0]}, index=full_index)
    for i in range(1, len(aList)):
        temp = pd.DataFrame({i: aList[i]}, index=full_index)
        windows = windows.merge(temp, left_index=True, right_index=True, how='left')
        windows.index.names = ['i', 'j']
    return windows





def get_classes2(classImage, name):
    try:
        with timer.Timer() as t:
            x = range(0, classImage.shape[1])
            y = range(0, classImage.shape[2])
            full_index = pd.MultiIndex.from_product([x, y], names=['i', 'j'])
            myfunc = lambda a: a.flatten()
            aList = [myfunc(classImage[i, :, :]) for i in range(0, classImage.shape[0])]
            classes = pd.DataFrame({name: aList[0]}, index=full_index)
    finally:
        print('get_classes2 Request took %.03f sec.' % t.interval)
    return classes



def get_classes(classImage, name):
    try:
        with timer.Timer() as t:
            clas_dict = {}
            shape = classImage.shape
            for i in range(classImage.shape[1]):
                for j in range(classImage.shape[2]):
                    clas_dict[(i, j)] = classImage[0, i, j]
            full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
            classes = pd.DataFrame({name: pd.Series(clas_dict)}, index=full_index)
    finally:
        print('get_classes Request took %.03f sec.' % t.interval)
    return classes


def gen_windows(array, n):
    """
    Parameters
    ----------
    array: np array
        Image from which to draw windows

    n: int
        width of moving window

    Returns
    -------
    windows: pandas dataframe
        df with ixj rows, with one column for every pixel values in nxn window
        of pixel i,j
    """
    shape = array.shape
    print('SHAPE:  ',shape)
    start = int((n - 1) / 2)
    end_i = shape[1] - start
    end_j = shape[2] - start
    x = range(1, shape[1] +1 )
    y = range(1, shape[2] + 1)
    tuples = list(it.product(x, y))
    win_dict = {}
    for i in range(start, end_i):
        for j in range(start, end_j):
            win_dict[(i, j)] = return_window(array, i, j, n)
    windows = pd.Series(win_dict)
    windows.index.names = ['i', 'j']
    index = windows.index
    windows = pd.DataFrame(windows.apply(lambda x: x.flatten()).values.tolist(), index=index)
    return (windows)

# concession = 'app_oki'
# print(concession)
# all_class_image = dh.get_landcover_class_image(concession)
# print(all_class_image)
# class_file = sorted(glob.glob(all_class_image))[0]
# # if(write_input_data):
# #    write_data_array(file_list[0],concession,'class',)
#
# all_class = rx.open_rasterio(class_file)
# # write_data_array(class_file, 'Class'+concession)
# y1 = get_classes(all_class.data, 'clas')
# y2 = get_classes2(all_class.data, 'clas')
# print(y1)
# print(y2)

a = np.zeros((5,3))
print(a)
a[0,1]=5
a[0,2]=5
a[2,2]=3
a[2,1]=3
a[3,2]=3
a[4,2]=3
print(a)
myFrame = pd.DataFrame(a)
#print('myframe.T.shape:'  , myFrame.mode(axis=0)[:,0])
print('myframe.T:'  , myFrame.T)
print(myFrame)
print(stat.mode(np.transpose(a)))
print(stat.mode(np.transpose(a), axis=0))
print(stat.mode(np.transpose(a), axis=1))

temp0  = pd.DataFrame(myFrame.mode(axis=0))
print('temp0.shape: ', temp0.shape)
print('temp0: ', temp0)
temp1 = (pd.DataFrame(myFrame.T.mode(axis=1))[0])#.astype(int)
if (isinstance(temp1,(np.ndarray))):
    print('ndarray')
if (isinstance(temp1,(pd.Series))):
    print('Series')
if (isinstance(temp1,(pd.DataFrame))):
    print('DATAFRAME')
print('temp1.shape: ', temp1.shape)
print('temp1: ', temp1)
x = temp1.astype(int)
print(x)
print(x.shape)
#for i in range(5):
    #a = np.append(a, i)
