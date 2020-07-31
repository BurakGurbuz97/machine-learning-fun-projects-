import NST_utils as N
import tensorflow as tf
import os
import scipy
from matplotlib.pyplot import imshow, axis, show
tf.reset_default_graph()

class Artist:
    def __init__(self, model_path = "pretrained-model/imagenet-vgg-verydeep-19.mat",
                 STYLE_LAYERS = [('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)]):
        """
        Artist class needs vgg19 pretrained-model which can be
        downloaded from  http://www.vlfeat.org/matconvnet/pretrained/
        """
        self.model = None
        self.model_path = model_path
        #inspiration weight according to layers
        self.STYLE_LAYERS = STYLE_LAYERS
        print("[Artist] Pretrained model is ready.")
        
    def show_image(self, img, read_from_file = True):
        """
        Prints given image, by default it will read from file
        """
        if read_from_file:
            image = scipy.misc.imread(img)
        axis("off")
        imshow(image)
        show()
        
    
    def _compute_content_cost(self,a_C, a_G):
        """
        Compute the content cost of generated and content image.
        Choose activations of one middle layer for best result.
        This cost function forces optimizer to change content of randomly initialized
        image to look smilar to content image  
        """
        m, n_H, n_W, n_C = a_G.get_shape().as_list()
        #unroll 3-d volume to 2-d matrix
        a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[n_H*n_W,n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[n_H*n_W,n_C]))
        return (1/(4*n_H*n_W*n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))
        
        
    def _gram_matrix(self,A):
        """
        Computes gram matrix for findings which filters fire together.
        """
        GA = tf.matmul(A,A, transpose_b = True)
        return GA
    
    def _compute_layer_style_cost(self,a_S, a_G):
        """
        Style cost for one layer.
        We want generated image to activate similar filter combinations with stlye image.
        This cost force optimizer to use style of the stlye image 
        while generating contents of generated image.
        """
        m, n_H, n_W, n_C =  a_G.get_shape().as_list()
        a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))
        GS = self._gram_matrix(a_S)
        GG = self._gram_matrix(a_G)
        return (1/(4 * (n_C**2) * ((n_H*n_W)**2) ) * tf.reduce_sum(tf.square(tf.subtract(GS, GG))))
    
    def _compute_style_cost(self, sess):
        """
        Computes style cost of multiply layers according to self.STYLE_LAYERS
        """
        J_style = 0 #total style cost
        for layer_name, coeff in self.STYLE_LAYERS:
            out = self.model[layer_name]      
            #a_S evaluated by the session
            a_S = sess.run(out) 
            #Note that this is just computational node
            #input will be assign to it since it will change by time
            a_G = out 
            J_style_layer = self._compute_layer_style_cost(a_S, a_G)
            J_style += coeff * J_style_layer
        return J_style
            
            
    def _total_cost(self,J_content, J_style, alpha = 20, beta = 40):
        """
        Computes the total cost 
        """
        return (alpha * J_content + beta * J_style)
    
    def generate_image(self,content_image_path, stlye_image_path,img_name, num_iterations = 200):
        if not os.path.exists("./output/" + img_name):
            os.makedirs("./output/" + img_name)
        #BUILDING PART
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        
        #Load Content image from path
        content_image = scipy.misc.imread(content_image_path)
        content_image = N.reshape_and_normalize_image(content_image)
        
        #Load Style image from path
        style_image = scipy.misc.imread(stlye_image_path)
        style_image = N.reshape_and_normalize_image(style_image)
        
        #Create random image seeded by content image for faster convergence
        generated_image = N.generate_noise_image(content_image)
        
        #Load model here to place it on graph
        self.model = N.load_vgg_model(self.model_path)
        sess.run(self.model['input'].assign(content_image))
        out = self.model['conv4_2'] 
        a_C = sess.run(out) #Hidden layers activations for content_image
        
        a_G = out #Node for generated_image
        J_content = self._compute_content_cost(a_C, a_G) #tensorflow graph
        
        sess.run(self.model['input'].assign(style_image))
        J_style = self._compute_style_cost(sess)
        
        J = self._total_cost(J_content= J_content, J_style = J_style)
        
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(J)
        
        #TRAINING PART
        sess.run(tf.global_variables_initializer())
        sess.run(self.model["input"].assign(generated_image))
        for i in range(num_iterations):
            
            sess.run(train_step)
            generated_image_new = sess.run(self.model["input"])
            if i%20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                # save current generated image in the "/output" directory
                N.save_image("./output/" + img_name +"/" + str(i) + ".jpg", generated_image_new)
                self.show_image("./output/" + img_name +"/" + str(i) + ".jpg")
        # save last generated image
        N.save_image('./output/'+ img_name + '/generated_image.png', generated_image_new)
        sess.close()
        return generated_image_new