


import numpy as np
from scipy.special import ndtri
from scipy.stats import norm
from scipy.linalg import sqrtm
import itertools
import math
import matplotlib.pyplot as plt
import sys

# def comp_detection_individual(class_0_mean,class_1_mean,variance,alpha):
#
#
#     #FUNCTIONAL USE -- THE CURRENT FUNCTION COMPUTES THE THRESHOLD AND THE CORRESPONDING THE BETA VALUES
#     threshold = []
#     detection_rate = []
#     inverse_alpha_complement = ndtri(1-alpha)
#     rows = np.size(class_1_mean,0)  #these are the features
#
#     #now compute the threshold correpsonding to the desired alpha value
#     for i in range(0,rows):
#         threshold.append(float(class_0_mean[i] + inverse_alpha_complement*math.sqrt(variance[i,i])))
#     #now compute the detection rate corresponding to the desired alpha value
#     for i in range(0,rows):
#         detection_rate.append(1-int(float(norm.cdf((threshold[i] - class_1_mean[i])/math.sqrt(variance[i,i])))*100)/100)
#
#
#
#     return detection_rate



def neyman_classifier(class_0_mean,class_1_mean,variance,alpha,feature_list,n):


    threshold = []
    detection_rate = []
    inverse_alpha_complement = ndtri(1 - alpha)  # in neyman pearson given alpha we need to find the inverse first
    rows = np.size(class_1_mean, 0)  # these are the features

    # the most important step here is to get the variance
    # step a -- get the sqrt of the variance matrix
    sub_variance = variance[:,feature_list]
    sub_variance = sub_variance[feature_list,:]
    inv_variance = np.linalg.inv(n*variance)
    sub_inv_variance = inv_variance[:,feature_list]
    sub_inv_variance = sub_inv_variance[feature_list,:]
    sub_mu_0 = class_0_mean[feature_list]
    sub_mu_1 = class_1_mean[feature_list]

    sub_mu_0 = n*sub_mu_0
    sub_mu_1 = n*sub_mu_1
    sub_variance = n*sub_variance
    # sub_inv_variance = n*sub_inv_variance


    w = np.matmul((sub_mu_1 - sub_mu_0).T, sub_inv_variance)
    mean_for_alpha = int(float(np.matmul(w, sub_mu_0) * 100)) / 100
    mean_for_beta = int(float(np.matmul(w, sub_mu_1)) * 100) / 100
    transformed_variance = np.matmul(w, sub_variance)
    transformed_variance = int(float(np.matmul(transformed_variance, w.T)) * 100) / 100
    std_deviation = math.sqrt(transformed_variance)

    threshold = int((mean_for_alpha + inverse_alpha_complement * std_deviation) * 100) / 100
    beta = 1 - int(norm.cdf((threshold - mean_for_beta) / std_deviation) * 100) / 100
    return beta
def bayes_classifier(class_0_mean,class_1_mean, variance,feature_list,n):

    sub_variance = variance[:,feature_list]
    sub_variance = sub_variance[feature_list,:]
    inv_variance = np.linalg.inv(variance)
    sub_inv_variance = inv_variance[:,feature_list]
    sub_inv_variance = sub_inv_variance[feature_list,:]
    sub_mu_0 = class_0_mean[feature_list]
    sub_mu_1 = class_1_mean[feature_list]

    sub_mu_0 = n*sub_mu_0
    sub_mu_1 = n*sub_mu_1
    sub_variance = n*sub_variance
    sub_inv_variance = n*sub_inv_variance


    w = np.matmul((sub_mu_1 - sub_mu_0).T, sub_inv_variance)

    transformed_expectation_0 = int(float(np.matmul(w, sub_mu_0)*100))/100
    transformed_expectation_1 = int(float(np.matmul(w, sub_mu_1))*100)/100
    transformed_variance = np.matmul(w,sub_variance)
    transformed_variance = int(float(np.matmul(transformed_variance,w.T))*100)/100
    SNR = int(float(np.matmul(w,sub_mu_1 + sub_mu_0)/2)*100)/100
    z0 = (SNR - transformed_expectation_0)/transformed_variance
    z1 = (SNR - transformed_expectation_1)/transformed_variance
    detection = int((1 - norm.cdf(z1))*1000)/1000
    missed_detection = int(( norm.cdf(z1))*1000)/1000
    false_alarm = int((1 - norm.cdf(z0))*1000)/1000
    benign = int((norm.cdf(z0))*1000)/1000
    return detection
def global_detector(class_0_mean, class_1_mean, variance, beta_list,alpha_list,global_alpha):




    #step 1: compute the a_list
    a_list = []
    x_values = []
    a_matrix = np.zeros((1,len(beta_list))) #the length of a matrix will be the length of beta the number of features in this case
    N = len(beta_list)
    one_zero_Permuation = [np.reshape(np.array(i), (N,1)) for i in itertools.product([0, 1], repeat=1* N)]
    for i,v in enumerate(beta_list):
        a_k = np.log(beta_list[i]/alpha_list[i]) + np.log((1 - alpha_list[i])/(1 - beta_list[i]))
        a_list.append(a_k)
        a_matrix[0,i] = a_k

    print("The A (a1,a2....) values that go in the summation equation (a1\delta1 + a2\delta2 .... are ", a_list)
    print("")
    #get all the xvalues in the plot
    for i in range(0,len(one_zero_Permuation)):
        x_values.append(float(np.matmul(a_matrix,one_zero_Permuation[i])))
    print("The corresponding xvalues..that is the potential values that come from the summation (a1\delta + s2\delta2 ...are: ",x_values)
    print("")




    #now it is time to compute the y value, this is the probability of something occuring
    #this is basically either alpha_k or (1 - alphak) for  a particular feature

    y_probability_list = []
    for i in range(0,len(one_zero_Permuation)):
        current_permuatation = one_zero_Permuation[i]

        for j in range(0,len(beta_list)):
            current_feature = j
            current_val = current_permuatation[j]
            if current_val == 1: #this is the false alarm case (p(delta = 1 | Ho))
                value_to_multiply = alpha_list[j]
            else:
                value_to_multiply = 1 - alpha_list[j]


            #this if statement is just to make the multiplication easier
            if current_feature == 0:
                multiplied = value_to_multiply
            else:
                multiplied*=value_to_multiply

        y_probability_list.append(multiplied)

    print("probability of delta under null hypothesis", y_probability_list)

    sum = 0

    current_point = -1
    for i in range(len(y_probability_list)-1,-1,-1): #iterate through the right side of the tail distribution
        sum += y_probability_list[i]

        if sum >= global_alpha:

            critical_point = i+1
            current_point = i
            break;




    cdf_to_current_point = 0
    cdf_to_critical_point = 0

    for i in range(current_point,len(y_probability_list)):
        cdf_to_current_point += y_probability_list[i]

    for i in range(critical_point,len(y_probability_list)):
        cdf_to_critical_point += y_probability_list[i]


    #DONT FORGET CORNER CASE
    if current_point == len(y_probability_list)-1 or current_point == -1:
        print("An alpha value does not exist")
        sys.exit(1)
    slope = (cdf_to_current_point - cdf_to_critical_point)/(x_values[current_point] - x_values[critical_point])
    global_threshold = x_values[critical_point] +(global_alpha - cdf_to_critical_point)/slope

    print("Here the global threshold is " + str(global_threshold))


    # print("The threshold determined is: " + str(global_threshold))
    #
    #
    #
    #now to do the same computation for y list

    y_probability_list_beta = []
    for i in range(0,len(one_zero_Permuation)):
        current_permuatation = one_zero_Permuation[i]

        for j in range(0,len(beta_list)): #iterating through the number of features
            current_feature = j
            current_val = current_permuatation[j]
            if current_val == 1:
                value_to_multiply = beta_list[j]
            else:
                value_to_multiply = 1 - beta_list[j]

            if current_feature == 0:
                multiplied = value_to_multiply
            else:
                multiplied*=value_to_multiply

        y_probability_list_beta.append(multiplied)

    box_y_fit_1 = []
    box_x_fit_1 = []
    print("probability of delta under alternate hypothesis is: ", y_probability_list_beta)
    print("")
    for i in range(0,len(x_values)-1):
        if x_values[i] <= global_threshold and x_values[i+1] >= global_threshold:
            critical_i = i+1

            box_x_fit_1.append(x_values[i])
            box_x_fit_1.append(x_values[i + 1])
            box_y_fit_1.append(y_probability_list_beta[i])
            box_y_fit_1.append(y_probability_list_beta[i+1])
            temp = i+1


    slope_beta = (box_y_fit_1[1] - box_y_fit_1[0])/(box_x_fit_1[1] - box_x_fit_1[0])
    detection_differential = slope_beta*(global_threshold - box_x_fit_1[0]) + box_y_fit_1[0]


    sum_so_far = 0
    for i in range(critical_i,len(y_probability_list_beta)):
        sum_so_far += y_probability_list_beta[i]



    detection_rate = sum_so_far +  detection_differential  #THIS IS THE FINAL THING WE ARE LOOKING FOR
    # for q in range(temp,len(y_probability_list_beta)):
    #     detection_rate += y_probability_list_beta[q]
    print("the detection rate utilizing decision_fusion is: {}".format(int(detection_rate*10000)/10000))



    ######################### PLOT under NULL HYPOTHESIS #####################################################
    markerline, stemlines, baseline = plt.stem(x_values,y_probability_list, '-.')

    # setting property of baseline with color red and linewidth 2
    plt.setp(baseline, color='r', linewidth=2)
    plt.plot(global_threshold,global_alpha, marker='o', markersize=7, color="black")
    plt.xlabel("Threshold_points")
    plt.ylabel("CDF")
    plt.title("Discrete CDF under Null Hypothesis")

    plt.annotate(
        "Threshold",
        xy=(global_threshold,0), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(
        "Alpha",
        xy=(0,global_alpha), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(
        "Point of Interest",
        xy=(global_threshold, global_alpha), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.75),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-1'))

    plt.show()

    ######################### PLOT under NULL HYPOTHESIS #####################################################
    markerline, stemlines, baseline = plt.stem(x_values, y_probability_list_beta, '-.')

    # setting property of baseline with color red and linewidth 2
    plt.setp(baseline, color='r', linewidth=2)
    plt.plot(global_threshold, detection_rate, marker='o', markersize=7, color="black")
    plt.xlabel("Threshold_points")
    plt.ylabel("CDF")
    plt.title("Discrete CDF under Alternate Hypothesis")

    plt.annotate(
        "Threshold",
        xy=(global_threshold, 0), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(
        "Beta",
        xy=(0, detection_rate), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(
        "Point of Interest",
        xy=(global_threshold, detection_rate), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.75),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-1'))

    plt.show()

    return detection_rate





def automate_global_fusion(class_0_mean,class_1_mean,variance,alpha,feature_partition,n,global_alpha,type='bayes'):


    type_mapping = {'neyman':1,'bayes':0}
    beta_list = []
    alpha_list = []
    if type_mapping[type] ==  1:  #this is Neyman
        for i in range(0,len(feature_partition)):
            beta = neyman_classifier(class_0_mean,class_1_mean,variance,alpha,feature_partition[i],n)
            beta_list.append(beta)
            alpha_list.append(alpha)

    elif type_mapping[type] == 0:
        for i in range(len(feature_partition)):
            beta = bayes_classifier(class_0_mean,class_1_mean,variance,feature_partition[i],n)
            # print(beta)
            beta_list.append(beta)
            alpha_list.append(1-beta) #since in bayes classifier alpha = 1 - beta
        print(alpha_list)
        print(beta_list)

    global_detector(class_0_mean,class_1_mean,variance,beta_list,alpha_list,global_alpha)




if __name__ == "__main__":


    #people get to choose the type of classifier that should be designed..bayes or neyman (0 or 1)..the feature partition as well
    class_0_mean = np.matrix([[0],[0]])
    class_1_mean = np.matrix([[1],[1]])
    variance = np.matrix([[1, .5], [.5, 4]])
    alpha = 0.05
    feature_partition = [[0],[1]]
    n = 1
    global_alpha = 0.109
    type = 0
    automate_global_fusion(class_0_mean, class_1_mean, variance, alpha, feature_partition, n, global_alpha,'bayes')
    # ##BUILDING A NEYMAN PEARSON DETECTOR USING DECISIONS FROM INDIVIDUAL FEATURES
    # beta_list_np = comp_detection_individual(likelihood_0_mean,likelihood_1_mean,variance,alpha)
    # alpha_list_np = [alpha]*len(beta_list_np)  #this is the alpha in the case of neyman pearson lemma
    #
    # print("Detection rate using indivudal features under NP criterion comes out to be: ", beta_list_np)
    # print("detection rate using feature fusion of all the features under Neyman Pearson is: {}".format(comp_detection_all(likelihood_0_mean,likelihood_1_mean, variance, alpha)))
    # global_alpha = 0.05
    # global_detector_neyman(likelihood_0_mean,likelihood_1_mean, variance, beta_list_np, alpha_list_np,global_alpha)


 #STEP 1...SOLVE threshold for each of the features present



    #BUILDING A GLOBAL BAYES DETECTOR USING DECISIONS FROM INDIVIDUAL FEATURES
    #In a bayesian classifier with equal priors probability of missed detection is equal to probability of false alarm
    #PFA = PMD -----> alpha = 1 - beta

    # alpha_list_bayes = beta_list_bayes  # from the equation in comment above


#ASSUMPTIONS MADE IN THE CODE
    #1. ALPHAS are the same for each of the feature partitions