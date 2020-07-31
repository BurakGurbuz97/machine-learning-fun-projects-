#This file checks if Artist's member functions runs correctly
import NST_net as Net
import tensorflow as tf
from termcolor import colored
import numpy as np

artist = Net.Artist()

#TEST FOR COMPUTE_CONTENT_COST
tf.reset_default_graph()
with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = artist._compute_content_cost(a_C, a_G)
    if abs(J_content.eval() - 6.76559) < 0.00001 :
        print(colored("[Artist_test] compute_content_cost(a_C, a_G) runs correctly!", "blue"))
    else:
        print(colored("[Artist_test] compute_content_cost(a_C, a_G) failed test!", "red"))

#TEST FOR GRAM_MATRIX  
tf.reset_default_graph()
with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = artist._gram_matrix(A)
    if abs(GA.eval()[0][0] - 6.42230511) < 0.00001 :
        print(colored("[Artist_test] gram_matrix(A) runs correctly!", "blue"))
    else:
        print(colored("[Artist_test] gram_matrix(A) failed test!", "red"))

#TEST FOR COMPUTE_LAYER_STYLE   
tf.reset_default_graph()
with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = artist._compute_layer_style_cost(a_S, a_G)
    if abs(J_style_layer.eval() - 9.19028) < 0.00001 :
        print(colored("[Artist_test] compute_layer_style_cost(a_S, a_G) runs correctly!", "blue"))
    else:
        print(colored("[Artist_test] compute_layer_style_cost(a_S, a_G) failed test!", "red"))

#TEST FOR TOTAL_COST
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = artist._total_cost(J_content, J_style)
    if abs(J - 35.34667875478276) < 0.00001 :
        print(colored("[Artist_test] total_cost(J_content, J_style) runs correctly!", "blue"))
    else:
        print(colored("[Artist_test] total_cost(J_content, J_style) failed test!", "red"))
        
