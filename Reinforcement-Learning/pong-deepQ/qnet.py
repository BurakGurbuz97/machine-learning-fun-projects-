import tensorflow as tf
class Model():
    """Estimates Q-Values using DNN"""
    
    def __init__(self,name="Model", valid_actions = [0,1,2,3,4,5], batch_size = 32, learning_rate = 0.001):
        #Some hyperparameters
        self.valid_actions = valid_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = name
        with tf.variable_scope(self.name):
            self._build_graph()
        
    def _build_graph(self):
        #Inputs
        
        self.X = tf.placeholder(shape=[None,84,84,4], dtype=tf.float32, name = "X")
        self.y = tf.placeholder(shape = [None], dtype=tf.float32, name = "y")
        self.action_index = tf.placeholder(shape = [None], dtype= tf.int32, name = "action_index")
        
        #Convolutional layers
        self.conv1 = tf.layers.conv2d(self.X,32,8,4,activation = tf.nn.relu)
        self.conv2 = tf.layers.conv2d(self.conv1,64,4,2,activation = tf.nn.relu)
        self.conv3 = tf.layers.conv2d(self.conv2, 64,3,1, activation = tf.nn.relu)
        
        #Fully connected net
        self.flat = tf.layers.flatten(self.conv3)
        self.dense1 = tf.layers.dense(self.flat, units=512,activation= tf.nn.relu)
        self.qvals = tf.layers.dense(self.dense1, len(self.valid_actions))
        
        # Get the predictions for the chosen actions only
        gather_indices = tf.range(self.batch_size) * tf.shape(self.qvals)[1] + self.action_index
        self.action_predictions = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        
        # Calcualte the loss
        self.loss = tf.losses.mean_squared_error(self.y, self.action_predictions)
        #self.loss = tf.reduce_mean(self.losses)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
    #Estimate qvals according to actions
    def estimate(self, sess, state):
        return sess.run(self.qvals, feed_dict = {self.X: state})
    
    #Optimize for given batch of state-action_index-target_qval
    def optimize(self, sess, state, action_index, target_qval):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict = {self.X: state, self.action_index: action_index,
                           self.y: target_qval})
        return loss
    
#Sync models
def copy_model_parameters(sess, model1, model2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(model1.name)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(model2.name)]
    e2_params = sorted(e2_params, key=lambda v: v.name)
    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)
    sess.run(update_ops)
    
    return len(update_ops)