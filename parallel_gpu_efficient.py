
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from keras.layers import Flatten
from keras.models import Model
from numpy import linalg as LA
import time
from joblib import Parallel, delayed

# def calculate_SI(featuremap,label, mode = 'dontCare' ):
#     tf.reset_default_graph()
#     [number, size] = featuremap.shape
#     # [number, size] = tf.shape(array)
#     array_plhdr = tf.placeholder(dtype=tf.float32, shape=[number, size])
#     array = tf.get_variable('array', [number, size])
#     # array = tf.convert_to_tensor(featuremap)
#     label = tf.convert_to_tensor(label)
#
#     delta = tf.Variable(np.zeros(number, dtype=np.float32))
#
#     def cond(count_true, i, iters):
#         return tf.less(i, iters)
#
#     def body(count_true, i, iters):
#         # my_label = tf.gather(label, my_index)
#         my_label = label[i]
#
#         difference = tf.subtract(array, array[i][:], name=None)
#
#         norm = tf.norm(difference, ord=2, axis=1)
#         # print('shape: ',tf.shape(norm))
#         delta_ = tf.scatter_update(delta, i, 10 ** 6)
#         norm = norm + delta_
#         delta_ = tf.scatter_update(delta, i, 0)
#
#         min_index_norm = tf.argmin(norm)
#
#
#         if mode == 'dontCare':
#             equal = tf.equal(label[min_index_norm], my_label)
#         else:
#             min_value_norm = tf.reduce_min(norm)
#             whh = tf.where(tf.equal(norm, min_value_norm))
#             x = tf.gather(label, whh)
#             equal_label = tf.equal(x, my_label)
#             if mode == 'hard':
#                 equal = tf.reduce_all(equal_label)
#             elif  mode == 'easy':
#                 equal = tf.reduce_any(equal_label)
#
#         count_true = tf.cond(equal, lambda: tf.add(count_true, 1),
#                              lambda: count_true)
#         return [count_true, tf.add(i, 1), iters]
#
#
#     count_true = tf.constant(0)
#     iters = tf.constant(number)
#     [result, _, _] = tf.while_loop(cond, body, [count_true, 0, iters])  # ,parallel_iterations=100
#     with tf.Session() as sess:
#         try:
#             sess.run(tf.global_variables_initializer())
#         except:
#             print("eeeeeeeeeeeeeeee")
#         sess.run(array.assign(array_plhdr), {array_plhdr: featuremap})
#         result_count = sess.run(result)
#         print(result_count)
#         # sess.close()
#         return float(result_count), float(number)

# def calculate_SIo(featuremap, label,mode='dontCare', flag=True):
#     tf.reset_default_graph()
#
#     def cond(i, size_loop):
#         return tf.less(i, size_loop)
#
#     def body(i, size_loop):
#         difference = tf.subtract(array, array[i][:], name=None)
#
#         norm = tf.norm(difference, ord=2, axis=1)
#         norm = tf.math.square(norm)
#
#         delta = tf.get_variable("delta", [number], dtype=tf.float32, initializer=tf.constant_initializer(0))
#         delta = tf.scatter_update(delta, i - 1, 0)
#         delta = tf.scatter_update(delta, i, 10 ** 3)
#         norm = tf.math.add(norm, delta)
#
#         a = tf.get_variable("a", [number, number], dtype=tf.float32, initializer=tf.constant_initializer(0))
#         a = tf.scatter_update(a, i, norm)
#
#         tf.get_variable_scope().reuse_variables()  # Reuse variables
#         with tf.control_dependencies([a]):
#             return (tf.add(i, 1), size_loop)
#
#     with tf.Session() as sess:
#         [number, size] = featuremap.shape
#         array_plhdr = tf.placeholder(dtype=tf.float32, shape=[number, size])
#         array = tf.get_variable('array', [number, size])
#
#         size_loop = tf.constant(number)
#         i = tf.constant(0)
#         i, _ = tf.while_loop(cond, body, [i, size_loop])  # ,parallel_iterations=100
#
#         a = tf.get_variable("a", [number, number], dtype=tf.float32)
#
#         sess.run(tf.initialize_all_variables())
#         sess.run(array.assign(array_plhdr), {array_plhdr: featuremap})
#         sess.run(i)
#         norm = sess.run(a)
#
#         if flag:
#             argmin = np.argmin(norm, axis=1)
#             label_min = label[argmin]
#             equal = (label_min == label)
#             ok = np.sum(equal)
#             return ok, number
#         else:
#             return norm

def calculate_SI(featuremap, label,mode='dontCare'):
    tf.reset_default_graph()
    # featuremap = tf.convert_to_tensor(featuremap)
    def cond(count_true,i, size_loop):
        return tf.less(i, size_loop)

    def body(count_true,i, size_loop):
        # difference = tf.subtract(array, array[i][:], name=None)
        # j = i.eval()
        # difference = tf.subtract(featuremap,featuremap)
        # norm = tf.norm(difference, ord=2, axis=1)
        # norm = tf.math.square(norm)

        # square = tf.math.reduce_sum(tf.math.square(array), axis=1)
        norm = tf.subtract(square, 2 * tf.tensordot(array, array[i, :], axes=1))

        delta = tf.get_variable("delta", [number], dtype=tf.float32, initializer=tf.constant_initializer(0))
        delta = tf.scatter_update(delta, i - 1, 0)
        delta = tf.scatter_update(delta, i, np.inf)
        norm = tf.math.add(norm, delta)

        min_index_norm = tf.argmin(norm)
        equal = tf.equal(label[min_index_norm], label[i])
        # equal = tf.constant(True)
        # if mode=='dintCare':
        #     equal = tf.equal(label[min_index_norm],  label[i])
        # else:
        #     min_value_norm = tf.reduce_min(norm)
        #     whh = tf.where(tf.equal(norm, min_value_norm))
        #     x = tf.gather(label, whh)
        #     equal_label = tf.equal(x, label[i])
        #     if mode == 'hard':
        #         equal = tf.reduce_all(equal_label)
        #     elif  mode == 'easy':
        #         equal = tf.reduce_any(equal_label)
        count_true = tf.cond(equal, lambda: tf.add(count_true, 1),
                             lambda: count_true)

        return (count_true,tf.add(i, 1), size_loop)

    with tf.Session() as sess:
        [number, size] = featuremap.shape
        array_plhdr = tf.placeholder(dtype=tf.float32, shape=[number, size])
        array = tf.get_variable('array', [number, size])
        label = tf.convert_to_tensor(label)

        square = tf.math.reduce_sum(tf.math.square(array), axis=1)

        size_loop = tf.constant(number)
        i = tf.constant(0)
        count_true = tf.constant(0)
        count_true, i ,_= tf.while_loop(cond, body, [count_true,i, size_loop])  # ,parallel_iterations=100

        sess.run(tf.initialize_all_variables())
        sess.run(array.assign(array_plhdr), {array_plhdr: featuremap})
        count,_ = sess.run([count_true,i])

        print(count)
        return count,number



# def calculate_SI_divide(featuremap, label, mode='dontCare'):
#     [number, size] = np.shape(featuremap)
#     norm_array1 = calculate_SI(featuremap[:, :int(size / 2)])
#     norm_array2 = calculate_SI(featuremap[:, int(size / 2):])
#     norm = np.add(norm_array1, norm_array2)
#     argmin = np.argmin(norm, axis=1)
#     label_min = label[argmin]
#     equal = (label_min == label)
#     ok = np.sum(equal)
#     return ok, number



def stability(featuremap,label,filename='result.txt',mode = 'dontCare'):
    def incremental(size = 50000):
        i = np.arange(1, 10)
        last = np.arange(1, 6)
        factor = [10, 100, 1000]
        x = [0]
        for element in factor:
            temp = element * i
            x = np.append(x, temp)
        temp = 10000 * last
        x = np.append(x, temp)
        return x

    output_file = open(filename, 'w', 1)
    # np.random.seed(0)
    index_inxremantal = incremental()
    result_array = np.zeros(len(index_inxremantal))
    precision_array = np.zeros(len(index_inxremantal))
    number_array = np.zeros(len(index_inxremantal))
    print("incremental " , index_inxremantal)

    featuremap, label = shuffle(featuremap, label, random_state=0)
    # featuremap, label = shuffle(featuremap, label)
    # print(featuremap[:index_inxremantal[2], :].shape)
    # print(label[:index_inxremantal[2]].shape)

    for i in np.arange(1, len(index_inxremantal)):
        # print("innnn")
        # result, number = calculate_SI(featuremap[:index_inxremantal[i], :],
        result, number = calculate_SI(featuremap[:index_inxremantal[i], :],
                                   label[:index_inxremantal[i]])
        precision = 1.0*result/number
        result_array[i], precision_array[i], number_array[i] = result, precision, number
        output_file.write("%i %f %f %i\n" % (i, result, precision, int(number)))
        print( i, "  ", result_array[i],"  ", precision_array[i],"  ", number )

    #
    output_file.close()
    return result_array,precision_array, number_array



def simple_SI(U, label,mode='dontCare'):
    count = 0
    for i in range(len(label)):
        # print("________________")
        difference = np.subtract(U, U[i][:])
        norm = LA.norm(difference, axis=1)
        norm[i] =  np.inf
        min_index_norm = np.argmin(norm)
        # print( i , 'norm = ', norm, min_index_norm)
        # print( np.sort(norm), label[min_index_norm], label[i],min_index_norm)
        equal = False
        if mode == 'dontCare':
            equal=(label[min_index_norm] == label[i])
        else:
            # min_value_norm = tf.reduce_min(norm)
            y = np.where(norm == norm[min_index_norm])
            arg_all = y[0]
            equal_label = (label[arg_all] == label[i])
            if mode == 'hard':
                equal = np.all(equal_label)
            elif  mode == 'easy':
                equal = np.any(equal_label)

        if equal:
            count+=1
        # print(min_index_norm)
    # print(count)
    # print(count, len(label))
    return count, len(label)



def parallel_cpu(U, Y):
    # print("shape U:", U.shape)
    [number, size] = U.shape

    # ok = 0
    # print("for")
    # d = np.zeros((number, size))
    # diff = np.zeros( (10000,10000, 65536),dtype=float)
    # t = time.time()
    def myfun(q):
        # my_label =
        difference = np.subtract(U, U[q][:])
        norm = LA.norm(difference, axis=1)
        norm[q] = np.inf
        min_index_norm = np.argmin(norm)
        if Y[min_index_norm] == Y[q]:
            return 1
        return 0

    x = np.arange(number)
    # results = Parallel(n_jobs=4, verbose=11, backend="threading")(
    results = Parallel(n_jobs=1, backend="threading")(
                 map(delayed(myfun),x))
    ok = np.sum(results)
    # for q in range(Q):
    #
    #
    #     if q % 1000 == 0:
    #         print('foe', q, Q)
    #     t = time.time()
    #     my = U[q][:]
    #     print('my shape', my.shape)
    #     my_label = Y[q]
    #     difference = np.subtract(U, my)
    #     print('subtract: ', time.time() - t, difference.shape)
    #
    #     # print("dif" , difference.shape)
    #     t = time.time()
    #     norm = LA.norm(difference, axis=1)
    #     print('norm: ', time.time() - t)
    #     print(norm.shape)
    #     t = time.time()
    #
    #     norm[my_label] = np.inf
    #     min_index_norm = np.argmin(norm)
    #     if Y[min_index_norm] == my_label:
    #         ok += 1
    #
    #
    #     print('min ', time.time() - t)
    #     print("_________")
    return ok, number



def calculate_SI_CPU(array , label, normalize=False):

    ok=0
    [number, size] = array.shape
    # array.astype(np.float16)
    squ = np.sum(np.square(array,),axis=1)
    # print("done")
    if normalize:
        # mean = np.mean(array,axis=0)
        # std = np.std(array,axis=0)
        # array = (array - mean) / (std + 1e-7)
        min = np.min(array, axis=0)
        max=  np.max(array, axis=0)
        array = (array-min)/(max-min+1e-7)
    # print(array.shape, min.shape)
    # print(np.mean(std),np.mean(mean),)
    for i in range(number):
        # print(i)
        if i % 1000==0:
            print(i)
        x = squ-2*np.inner(array, array[i, :])
        x[i] = np.inf
        minIndex= np.argmin(x)
        if label[minIndex] == label[i]:
            ok += 1
    return ok, number

def calculate_SI2(featuremap, label,mode='dontCare'):
    tf.reset_default_graph()
    # featuremap = tf.convert_to_tensor(featuremap)
    def cond(count_true,i, size_loop):
        return tf.less(i, size_loop)

    def body(count_true,i, size_loop):
        # difference = tf.subtract(array, array[i][:], name=None)
        # j = i.eval()
        # difference = tf.subtract(featuremap,featuremap)
        # norm = tf.norm(difference, ord=2, axis=1)
        # norm = tf.math.square(norm)

        # square = tf.math.reduce_sum(tf.math.square(array), axis=1)
        norm = tf.subtract(square, 2 * tf.tensordot(array, array[i, :], axes=1))

        delta = tf.get_variable("delta", [number], dtype=tf.float32, initializer=tf.constant_initializer(0))
        delta = tf.scatter_update(delta, i - 1, 0)
        delta = tf.scatter_update(delta, i, np.inf)
        norm = tf.math.add(norm, delta)

        min_index_norm = tf.argmin(norm)
        equal = tf.equal(label[min_index_norm], label[i])

        count_true = tf.cond(equal, lambda: tf.add(count_true, 1),
                             lambda: count_true)

        return (count_true,tf.add(i, 1), size_loop)

    with tf.Session() as sess:
        [number, size] = featuremap.shape
        array_plhdr = tf.placeholder(dtype=tf.float32, shape=[number, size])
        array = tf.get_variable('array', [number, size])
        label = tf.convert_to_tensor(label)

        square = tf.math.reduce_sum(tf.math.square(array), axis=1)

        size_loop = tf.constant(number)
        i = tf.constant(0)
        count_true = tf.constant(0)
        count_true, i ,_= tf.while_loop(cond, body, [count_true,i, size_loop])  # ,parallel_iterations=100

        sess.run(tf.initialize_all_variables())
        sess.run(array.assign(array_plhdr), {array_plhdr: featuremap})
        count,_ = sess.run([count_true,i])

        print(count)
        return count,number


# x = np.asarray([ 3.5, 4,5,6])
# x2 = np.asarray([6,5,6,7])
# y = np.asarray([x, x2, 2+x, x2+1, 3+x, x2-1],dtype=np.float32)
# label = np.asarray([1, 0, 0,2,1,1])
#
# print(y.shape)
# print(label.shape)
#
# b,a= calculate_SI(y,label)
# print(b,a)
