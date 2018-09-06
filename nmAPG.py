import numpy as np

number_of_user = 100
number_of_song = 50
number_of_metapath = 5
rank_of_latent_factor = 2
rank_of_correlation = 2

sigma_w = 0.1126
sigma_V = 0.1126


def prox_pre_w(w, V):
    """need rank_of_latent factor, number_of_metapath, alpha"""
    sub_w_list = []
    index_list = []
    index = 0
    while index + rank_of_latent_factor <= 2 * number_of_metapath * rank_of_latent_factor:
        w_part = w[index:index + rank_of_latent_factor]
        index_list.append(index)
        sub_w_list.append(w_part)
        index += rank_of_latent_factor
    gradient_list = make_gradient_w(index_list, w, V)
    for i in range(len(sub_w_list)):
        sub_w_list[i] -= alpha*gradient_list[i]
    return sub_w_list

def make_gradient_w(index_list, w, V):
    gradient_list = [0 for i in range(len(index_list))]
    for i in range(len(user_item_pair)):
        prediction = y(w, V, user_item_pair[i])
        for x in range(len(index_list)):
            x_part = user_item_pair[i][index_list[x]:index_list[x] + rank_of_latent_factor]
            gradient_list[x] += 2 * (rating[i] - prediction) * x_part
    for j in range(len(index_list)):
        gradient_list[j] += sigma_w * np.ones(len(x_part))
    return gradient_list

def prox_pre_V(w,V):
    sub_V_list = []
    index_list = []
    index = 0
    while index + rank_of_latent_factor <= 2 * number_of_metapath * rank_of_latent_factor:
        V_part = V[index:index + rank_of_latent_factor]
        index_list.append(index)
        sub_V_list.append(V_part)
        index += rank_of_latent_factor
    gradient_list = make_gradient_V(index_list, w, V)
    for i in range(len(sub_V_list)):
        sub_V_list[i] -= alpha * gradient_list[i]
    return sub_V_list

def make_gradient_V(index_list, w, V):
    gradient_list = [0 for i in range(len(index_list))]
    for i in range(len(user_item_pair)):
        prediction = y(w, V, user_item_pair[i])
        for x in range(len(index_list)):
            V_part = compute_gradient_V(index_list[x], user_item_pair[i], V)
            gradient_list[x] += 2 * (rating[i] - prediction) * V_part
    for j in range(len(index_list)):
        gradient_list[j] += sigma_V*np.ones(np.shape(V_part))
    return gradient_list

def h(w, V):
    """need user_item_pair, rating, sigma_w, sigma_V"""
    result = 0
    for i in range(number_of_song*number_of_user):
       result += (rating[i] - y(w, V, user_item_pair[i]))**2
    index = 0
    while index + rank_of_latent_factor<= 2*number_of_metapath*rank_of_latent_factor:
        w_part = w[index:index+rank_of_latent_factor]
        V_part = V[index:index+rank_of_latent_factor]
        index += rank_of_latent_factor
        result += sigma_w*np.linalg.norm(w_part) + sigma_V*np.linalg.norm(V_part)
    return result

def y(w, V, x):
    result = 0
    result += np.dot(w, x)
    length = len(x)
    for i in range(length):
        for j in range(i + 1, length):
            result += np.dot(V[i], V[j])*x[i]*x[j]
    return result


def compute_gradient_V(index,sample, V):
    """need rank_of_latent_factor, number_of_metapath"""
    V_ig = np.zeros((rank_of_latent_factor, rank_of_correlation))
    for x in range(rank_of_latent_factor):
        V_part = np.zeros(rank_of_correlation)
        for i in range(2*number_of_metapath*rank_of_latent_factor):
            if i == index+x:
                continue
            V_part += V[i]*sample[i]*sample[index+x]
        V_ig[x] = (V_part)
    return V_ig

def proximal_w(array, sigma):
    """need nothing"""
    result = []
    for i in range(len(array)):
        result_ig = max(1-sigma/np.linalg.norm(array[i]), 0)*array[i]
        result = np.append(result, result_ig)
    return result

def proximal_V(array, sigma):
    """need nothing"""
    result = []
    for i in range(len(array)):
        result_ig = max(1-sigma/np.linalg.norm(array[i]), 0)*array[i]
        if len(result) == 0:
            result = result_ig
        else:
            result = np.vstack((result, result_ig))
    return result




#initialization
user_item_pair = []
rating = []
for x in range(number_of_user*number_of_song):
    user_item_pair.append(np.random.rand(2*number_of_metapath*rank_of_latent_factor))
    rating.append(np.random.rand())

user_item_pair = np.asarray(user_item_pair)


#nmAPG
w_0 = np.random.rand(2*number_of_metapath*rank_of_latent_factor)
V_0 = np.random.rand(2*number_of_metapath*rank_of_latent_factor, rank_of_correlation)
w_t = w_level = w_old = w_0
V_t = V_level = V_old = V_0
c_t = c_old = h(w_t, V_t)
q_t = q_old = 1
var_1 = 10 ** (-3)
var_2 = 0.1126
a_old = 0
a_t = 1
alpha = 10**(-7)
for i in range(10):
    y_t = w_t + a_old/a_t*(w_level - w_t) + (a_old - 1)/a_t*(w_t - w_old)
    Y_t = V_t + a_old/a_t*(V_level - V_t) + (a_old - 1)/a_t*(V_t - V_old)
    w_level = proximal_w(prox_pre_w(y_t, Y_t), sigma_w)
    V_level = proximal_V(prox_pre_V(y_t, Y_t), sigma_V)
    delta_t = np.linalg.norm(w_level - y_t)**2 + np.linalg.norm(V_level - Y_t)**2
    h_level = h(w_level, V_level)
    if h_level <= c_t - var_1*delta_t:
        w_t = w_level
        V_t = V_level
        h_t = h_level
    else:
        w_top = proximal_w(prox_pre_w(w_t, V_t), sigma_w)
        V_top = proximal_V(prox_pre_V(w_t, V_t), sigma_V)
        h_top = h(w_top, V_top)
        if h_top < h_level:
            w_t = w_top
            V_t = V_top
            h_t = h_top
        else:
            w_t = w_level
            V_t = V_level
            h_t = h_level
    a_old = a_t
    a_t = 1/2*(np.sqrt(4*a_t**2 + 1) + 1)
    q_old = q_t
    q_t = var_2*q_t + 1
    c_old = c_t
    c_t = 1/q_t*(var_2*q_old*c_old + h_t)

print(w_t)
print(V_t)



