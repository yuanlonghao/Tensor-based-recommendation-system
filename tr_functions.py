#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy.linalg as la
import timeit
"""
TR functions
"""
# generate tr core tensors randomly
def init_tr_cores(tensor_size, tr_rank, value = 'random'):
    tr_cores = []
    ndims = len(tensor_size)
    # print(ndims)
    tr_rank.append(tr_rank[0])
  #  print(tr_rank)
    if value == 'random':
            for n in range(0, ndims):
                tr_cores.append(0.1 * np.random.rand(tr_rank[n], tensor_size[n], tr_rank[n+1]))
    elif value == 'zeros':
            for n in range(0, ndims):
                tr_cores.append( np.zeros((tr_rank[n], tensor_size[n], tr_rank[n+1])))
    elif value == 'large_value':
            for n in range(0, ndims):
                tr_cores.append(0.0005 * np.random.rand(tr_rank[n], tensor_size[n], tr_rank[n+1]))
        # print(len(tr_cores))
    return tr_cores

# merge tr_cores EXCEPT the nth core
# important operatition of TRD
def core_merge(tr_cores, n):
    dim = len(tr_cores)
   # print(dim)
    tr_cores_shift = tr_cores[n:dim] + tr_cores[0:n] # shift the nth core to the last
    #for i in range(3):
       # print(tr_cores_shift[i].shape)
    tr_mul = np.copy(tr_cores_shift[0])
    for i in range(dim-2):
       # print(i)
        temp_core = np.copy(tr_cores_shift[i+1])
        zl = tr_mul.reshape(int(tr_mul.size/temp_core.shape[0]), temp_core.shape[0],  order = 'F').copy()
        zr = temp_core.reshape(temp_core.shape[0], temp_core.shape[1] * temp_core.shape[2],  order = 'F').copy()
        tr_mul = np.dot(zl, zr)
    s1 = tr_cores_shift[0].shape[0]
    s2 = tr_cores_shift[dim-2].shape[2]
    merge_neq_out = tr_mul.reshape(s1, int(tr_mul.size/(s1 * s2)), s2,  order = 'F').copy()
    return merge_neq_out

# convert core tensors to the approximated tensor (a simple but high computational cost way)
def cores2tensor(tr_cores):
    dim = len(tr_cores)
    tensor_size = []
    for i in range(dim):
        tensor_size.append(tr_cores[i].shape[1])
    # print(tensor_size)
    # merge all the cores 
    tr_mul = tr_cores[0]
    for i in range(dim-1):
        temp_core = tr_cores[i+1]
        zl = tr_mul.reshape(int(tr_mul.size/temp_core.shape[0]), temp_core.shape[0], order = 'F').copy()
        zr = temp_core.reshape(temp_core.shape[0], temp_core.shape[1] * temp_core.shape[2], order = 'F').copy()
        tr_mul = np.dot(zl, zr)
    s = tr_cores[0].shape[0]
    core_merge = tr_mul.reshape(s, np.prod(tensor_size), s, order = 'F').copy()
    #print(core_merge.shape)
    temp1 = core_merge.transpose(1, 2, 0).reshape(np.prod(tensor_size), s * s, order = 'F').copy()
    temp2 = np.eye(s, s).reshape(s * s, 1, order = 'F').copy()
    tensor_approx = np.dot(temp1, temp2).reshape(tensor_size, order = 'F').copy()
    return tensor_approx

# Different way to reconstruct the tensor
def cores2tensor_new(tr_cores):
    dim = len(tr_cores)
    tensor_size = []
    for i in range(dim):
        tensor_size.append(tr_cores[i].shape[1])
    n = 1
    G_neq = core_merge(tr_cores, n)
    G_neq_mat = tensor2mat(G_neq, 2, mat_type=3)
    mat = np.dot(tensor2mat(tr_cores[n-1], 2, mat_type=1), G_neq_mat.T)
    tensor = mat2tensor(mat, n, tensor_size, mat_type = 3)
    return tensor

# reshape tensor to matrix
'''
input_tensor size:  I_1 x I_2 x ... I_N 
mat_type=1: kolda matricization → I_n x I_1I_2...I_n-1I_n+1...I_N
mat_type=2: tensor train matricization → I_1I_2...I_n x I_n+1...I_N
mat_type=3: tensor ring matricization → I_n x I_n+1...I_NI_1...I_n-1
'''
def tensor2mat(input_tensor, n, mat_type=1):
    tensor_size = input_tensor.shape
    num = input_tensor.size
    dim = len(tensor_size)
    if mat_type == 1:
        arr = np.append(n - 1, np.arange(0, n - 1))
        arr = np.append(arr, np.arange(n, dim))
        temp = input_tensor.transpose(arr)
        mat = temp.reshape(tensor_size[n-1], int(num/tensor_size[n-1]), order = 'F').copy()
    elif mat_type ==2:
        arr = np.append(np.arange(0, n), np.arange(n, dim))
        temp = input_tensor
        mat = temp.reshape(np.prod(tensor_size[0:n]), np.prod(tensor_size[n:dim]), order = 'F').copy()
    elif mat_type ==3:
        arr = np.append(np.arange(n - 1, dim), np.arange(0, n - 1))
        temp = input_tensor.transpose(arr)
        mat = temp.reshape(tensor_size[n-1], int(num/tensor_size[n-1]), order = 'F').copy()
    # print("Type: %d" %(mat_type), ", Reshape at mode-%d" %(n), ", Transpose index:", arr, ", Matrix size: %u x %u" %(mat.shape[0], mat.shape[1]))
    return mat

# reshape the "matricized tensor" to tensor
def mat2tensor(input_matrix, n, tensor_size, mat_type=1):
    dim = len(tensor_size)
    if mat_type == 1:
        arr = np.append(tensor_size[n-1], int(np.prod(tensor_size[0:n-1])))
        arr = np.append(arr, int(np.prod(tensor_size[n: dim])))
        temp = input_matrix.reshape(arr, order = 'F').transpose(1, 0, 2).copy()
        output_tensor = temp.reshape(tensor_size, order = 'F').copy()
    elif mat_type == 2:
        output_tensor = input_matrix.reshape(tensor_size, order = 'F').copy()
    elif mat_type == 3:
        arr = np.append(int(np.prod(tensor_size[n-1:dim])), int(np.prod(tensor_size[0:n - 1])))
        temp = input_matrix.reshape(arr, order = 'F').transpose(1, 0).copy()
        output_tensor = temp.reshape(tensor_size, order = 'F').copy()
    # print("The size of tensor is", output_tensor.shape)
    return output_tensor


# Tensor matrix product Y = X x_n U  ---> Y_(n) = U x X_(n)
def tensor_matrix_product(tensor, matrix, n):
    tensor_size = list(tensor.shape)
    dim = len(tensor_size)
    tensor_n = tensor2mat(tensor, n , 1)
    tensor_product_n = np.dot(matrix, tensor_n)
    tensor_size[n-1] = matrix.shape[0]
    tensor_product = mat2tensor(tensor_product_n, n, tensor_size, 1)
    return tensor_product

# To calculate the multi-product of the core slices
# remove_n = None --> for calculating the tr tensor value w.r.t. the index 
# remove_n = n --> for calculating the gradient (order: n+1 -> N-> 1-> n-1)
def cores_multi(cores, index, remove_n = None):
    index = list(index)
    dim = len(index)
    #print(index, dim)
    if remove_n == None:
        temp = cores[0][:,index[0],:]
        for i in range(1, dim):
            temp = np.dot(temp, cores[i][:,int(index[i]),:])
    else:
        #print('remove_n is', remove_n, 'dim is', dim, 'index is',index)
        index_re =  index[remove_n:dim] + index[0:remove_n]
        #print('index_re',index_re)
        cores_re =  cores[remove_n:dim] + cores[0:remove_n]
        temp = cores_re[0][:,index_re[0],:]
        for i in range(1, dim - 1):
            temp =  np.dot(temp, cores_re[i][:,index_re[i],:]) # matrix size: R_n+1  x  R_n
    return temp

# To pick out all the indices and values of non-zero entries in a tensor
# Out put X-train --> sample x index   y_train --> values
def tensor2sptensor(tensor):
    X_train = np.argwhere(tensor != 0)
    y_train = []
    for i in range(X_train.shape[0]):
        y_train.append(tensor[tuple(X_train[i])])
    return np.array(X_train), np.array(y_train)

# To evaluate the performance of the tensor completion
# RMSE: root mean squared error
def evaluation_RMSE(tensor_real, tensor_eval):
    num = tensor_real.size
    err_sum = np.square(tensor_real - tensor_eval).sum()
    RMSE = np.sqrt(err_sum / num)
    return RMSE

def evaluation_RSE(tensor_real, tensor_eval):
    err_sum = np.square(tensor_real - tensor_eval).sum()
    tensor_real_sum = np.square(tensor_real).sum()
    RSE = np.sqrt(err_sum /tensor_real_sum)
    return RSE

def evaluation_MAE(tensor_real, tensor_eval):
    err = np.subtract(tensor_real, tensor_eval)
    MAE = np.abs(err).sum()/tensor_real.size
    return MAE

# Generate a binary tensor in which 0 and 1 stand for missing entries and obsverations respectively
def generate_index_tensor(tensor_size, missing_rate):
    num = int(np.prod(tensor_size))
    num_missing = int(np.ceil(num * missing_rate))
    index = np.append(np.zeros(num_missing), np.ones(num - num_missing))
    index_shuffle = np.array(np.random.permutation(index))
    index_tensor = np.copy(index_shuffle.reshape(tensor_size, order = 'F'))
    return index_tensor



# tensor ring alternating least square
def TR_ALS(input_tensor, tr_rank, maxiter=10):
    tensor_size = input_tensor.shape
    dim = len(tensor_size)
    tr_cores = init_tr_cores(tensor_size, tr_rank)
    print('Converging TR-ALS')
    for i in range(maxiter):
        for n in range(1, dim+1):
           # print('n=', n)
            core_merge_flatten_trans = np.transpose(tensor2mat(core_merge(tr_cores, n), 2, mat_type=3))
            G_neq_pinv =  np.linalg.pinv(core_merge_flatten_trans)
            tr_cores[n-1] = mat2tensor(np.dot(tensor2mat(input_tensor, n , mat_type = 3), G_neq_pinv),2,tr_cores[n-1].shape,mat_type=1)
        print('.' ,end='')
    print('Finished!')
    return tr_cores



# TR-ALS-EM (expectation maxmization method)
# Able to deal with incomplete tensor
# Merit: no tuning parameters, fast convergence
def TR_ALS_EM(input_tensor, missing_index, tr_rank, maxiter=10):
    tensor_size = input_tensor.shape
    dim = len(tensor_size)
    tr_cores = init_tr_cores(tensor_size, tr_rank)
    missing_index_re = np.ones(tuple(tensor_size)) - missing_index
    tensor = input_tensor * missing_index + cores2tensor_new(tr_cores) * missing_index_re
    for i in range(maxiter):
        for n in range(1,dim+1):
            core_merge_flatten_trans = np.transpose(tensor2mat(core_merge(tr_cores, n), 2, mat_type=3))
            G_neq_pinv =  np.linalg.pinv(core_merge_flatten_trans)
            tr_cores[n-1] = mat2tensor(np.dot(tensor2mat(tensor, n , mat_type = 3), G_neq_pinv),2,tr_cores[n-1].shape,mat_type=1)
        tensor_appr = cores2tensor(tr_cores)
        tensor = input_tensor * missing_index + tensor_appr * missing_index_re
        print('Iteration', i+1,'finished, RSE is', evaluation_RSE(input_tensor * missing_index, tensor_appr * missing_index))
    return tensor, tr_cores
            

# Project a large-scale tensor into a small-sized tensor which owns the most 'actions' of the large-scale tensor
def tensor_projection(tensor, projection_size, p = 2, q = 2):
    dim = len(tensor.shape)
    P = tensor.copy()
    Q = []
    for n in range(1, dim+1):
        size = P.shape
        p_size_n = projection_size[n-1] + q
        projection_matrix = np.random.rand(int(np.array(size).prod()/size[n-1]), p_size_n)
        Pn = tensor2mat(P, n, 1) 
        Y = np.dot(Pn, projection_matrix)
        for j in range(p):
            _, L1, _ = la.lu(Y)
            _, L2, _ = la.lu(np.dot(Pn.transpose(), L1))
            Y = np.dot(Pn, L2)
        Qn, _ = la.qr(Y, mode = 'economic')
        Q.append(Qn)
        P = tensor_matrix_product(P, Qn.transpose(), n)
    return P, Q

# back projection of the tr cores of the projected small-sized tensor to the tr cores of large-scale tensor
def tensor_back_projection(tr_cores, Q):
    dim = len(tr_cores)
    tr_cores_approx = []
    for n in range(1, dim+1):
        core_n = tensor_matrix_product(tr_cores[n-1], Q[n-1], 2)
        tr_cores_approx.append(core_n)
    return tr_cores_approx

# Do TR-SGD in a straight forward way
def TR_SGD(X_train, y_train, tensor_size, tr_rank, learning_rate = 0.01, epoch = 1, alpha = 0.001, beta = 1):
    # initialization
    dim = X_train.shape[1]
    #tensor_size = X_train.max(axis = 0) + 1
    #print('tensor_size is',tensor_size)
    cores = init_tr_cores(tensor_size, tr_rank, value = 'random')
    for i in range(len(cores)):
        cores[i] = beta * cores[i]
        
    # calculate the gradient
    for epoch in range(epoch):
        shuffle_list = np.random.permutation(y_train.shape[0])
        #print('order is ',shuffle_list)
        for i in shuffle_list:
            #print('sample', i)
            err = np.trace(cores_multi(cores, X_train[i], remove_n = None)) - y_train[i]
            #print(err)
            for n in range(dim):
                temp = cores_multi(cores, X_train[i], n+1)
                cores[n][:,X_train[i,n],:] = cores[n][:,X_train[i,n],:] - learning_rate * (err * temp.T + alpha * cores[n][:,X_train[i,n],:])
                #print('update finish')
        y_approx_100 = []
        for k in range(100):
            y_approx_100.append(np.trace(cores_multi(cores, np.array(X_train[shuffle_list[k]]), remove_n = None)))
        RSE = evaluation_RSE(np.array(y_approx_100),np.array(y_train[shuffle_list[0:100]]))
        print('epoch', epoch + 1 , 'finish.','RSE of random 100 samples is',RSE)
    return cores

# Do TR-SGD in a straight forward way
def TR_SGD(X_train, y_train, tensor_size, tr_rank, learning_rate = 0.01, epoch = 1, alpha = 0.001, beta = 1):
    # initialization
    dim = X_train.shape[1]
    #tensor_size = X_train.max(axis = 0) + 1
    #print('tensor_size is',tensor_size)
    cores = init_tr_cores(tensor_size, tr_rank, value = 'random')
    for i in range(len(cores)):
        cores[i] = beta * cores[i]
        
    # calculate the gradient
    for epoch in range(epoch):
        shuffle_list = np.random.permutation(y_train.shape[0])
        #print('order is ',shuffle_list)
        for i in shuffle_list:
            #print('sample', i)
            err = np.trace(cores_multi(cores, X_train[i], remove_n = None)) - y_train[i]
            #print(err)
            for n in range(dim):
                temp = cores_multi(cores, X_train[i], n+1)
                cores[n][:,X_train[i,n],:] = cores[n][:,X_train[i,n],:] - learning_rate * (err * temp.T + alpha * cores[n][:,X_train[i,n],:])
                #print('update finish')
        y_approx_100 = []
        for k in range(100):
            y_approx_100.append(np.trace(cores_multi(cores, np.array(X_train[shuffle_list[k]]), remove_n = None)))
        RSE = evaluation_RSE(np.array(y_approx_100),np.array(y_train[shuffle_list[0:100]]))
        print('epoch', epoch + 1 , 'finish.','RSE of random 100 samples is',RSE)
    return cores


"""
import numpy as np
# Do TR-SGD with Adam gradient descent scheme
def TR_SGD_Adam(input_tensor, index_tensor, tr_rank, learning_rate = 0.01, batchsize=1, maxiter = 1000, tolerance = 0.03):
    # initialization
    beta1 = 0.9
    beta2 = 0.999
    m_t = init_tr_cores(tensor_size, tr_rank, value = 'zeros')
    v_t = init_tr_cores(tensor_size, tr_rank, value = 'zeros')
    
    tensor_size = input_tensor.shape
    dim = len(tensor_size)
    
    core_tensors = init_tr_cores(tensor_size, tr_rank, value = 'random')
    core_gradient =  init_tr_cores(tensor_size, tr_rank, value = 'zeros')
    
    tensor = np.multiply(input_tensor, index_tensor)
    index_all = np.argwhere(index_tensor != 0) # the index of the observed entries
    
    
    # calculate the gradient
    for it in range(maxiter):
        index_all_shuffle = np.random.permutation(index_all)
        index_batch = index_all_shuffle[0:batchsize,:] 
        for i in range(batchsize):
            for n in range(dim):
"""


