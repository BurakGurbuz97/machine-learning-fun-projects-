import tensorflow as tf
import gym
import numpy as np

#Calculate Discounted rewards for steps
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

#Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4
env = gym.make("CartPole-v0")
env._max_episode_steps = 500

#NeuralNet
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None,env.observation_space.shape[0]], name = "X")
mask_choice = tf.placeholder(tf.int32, shape=[None,2])

qvals = tf.placeholder(tf.float32, shape=[None], name = "qvals")
actions = tf.placeholder(tf.float32, shape=[None], name = "actions")

dense1 = tf.layers.dense(X, units=128, activation=tf.nn.relu, name = "dense1")
logits = tf.layers.dense(dense1, units=env.action_space.n, name = "logits")

softmax = tf.nn.softmax(logits)

choisen_softmax = tf.gather_nd(tf.nn.log_softmax(logits),mask_choice)

PG = tf.multiply(qvals, choisen_softmax)
loss = tf.reduce_mean(-PG)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    done_episodes = 0
    while(True):
        batch_states, batch_actions, batch_qvals = [], [], []
        total_rewards = []
        for episode in range(EPISODES_TO_TRAIN):    
            cur_rewards = []
            episode_rew = 0
            state = env.reset()
            while(True):
                probs = sess.run(softmax, feed_dict={X: [state]})[0]
                normaled_probs = probs / probs.sum()
                action = np.random.choice([0,1],p = normaled_probs)
                next_state, reward, done, _ =  env.step(action)
                episode_rew += reward
                batch_states.append(state)
                batch_actions.append(action)
                cur_rewards.append(reward)
                if done:
                    batch_qvals.extend(calc_qvals(cur_rewards))
                    total_rewards.append(episode_rew)
                    done_episodes +=1
                    #print("[INFO] done episode:{} total reward for this episode:{}".format(done_episodes,episode_rew))
                    break
                state = next_state
        mask = np.stack([np.arange(0,len(batch_actions)),batch_actions],axis=1)
        tot = float(np.mean(total_rewards[-100:]))
        print("episode: {} mean_100: {}".format(done_episodes,tot))
        if tot > 450:
            saver.save(sess, "./models/cartpole")
            break
        sess.run(optimizer, feed_dict={X: batch_states, qvals: batch_qvals, actions: batch_actions, mask_choice:mask})
             
             
with tf.Session() as sess:
    from gym.wrappers import Monitor
    saver.restore(sess,"./models/cartpole")
    env = gym.make("CartPole-v0")
    env._max_episode_steps = 500
    env = Monitor(env, directory="./monitor", force=True)
    state = env.reset()
    while(True):
        probs = sess.run(softmax, feed_dict={X: [state]})[0]
        normaled_probs = probs / probs.sum()
        action = np.random.choice([0,1],p = normaled_probs)
        next_state, reward, done, _ =  env.step(action)
        if done:
            break
        state = next_state
        
              
        
        
        

