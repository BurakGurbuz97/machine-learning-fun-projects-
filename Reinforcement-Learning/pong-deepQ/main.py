import time
import numpy as np
import qnet
import wrap_maxim
import collections
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()

def show_img(img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
    

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 16.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000
SAVE_EPISODE = 10
EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 0.02
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
               
def calculate_label_qvals(sess, batch, target_net):
    states, actions, rewards, dones, next_states = batch
    next_state_qvals = np.max(target_net.estimate(sess,next_states), axis = 1)
    for i in range(len(dones)):
        if dones[i] == True:
            next_state_qvals[i] = 0
            
    labels = next_state_qvals * GAMMA + rewards
    return labels

if __name__ == "__main__":
    env = wrap_maxim.make_env(DEFAULT_ENV_NAME)
    net = qnet.Model(name="net",batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    tgt_net = qnet.Model(name="tgt_net", batch_size=BATCH_SIZE,  learning_rate=LEARNING_RATE)
    exp_buf = ExperienceBuffer(REPLAY_SIZE)
    epsilon = EPSILON_START
    episode_counter = 0
    frame_id = 0
    total_reward = []
    state = env.reset()
    state = np.swapaxes(state, 0 ,2)
    state = np.swapaxes(state,0,1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/weigts")
        episode_reward = 0
        loss = 999
        start_t = time.time()
        start_f = frame_id
        while(True):
            frame_id += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_id / EPSILON_DECAY_LAST_FRAME)
            action = None
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = net.estimate(sess, [state])[0]
                action = np.argmax(q_vals)
                
            new_state, reward, done, _ = env.step(action)
            new_state = np.swapaxes(new_state, 0 ,2)
            new_state = np.swapaxes(new_state,0,1)
            
            episode_reward += reward
            exp = Experience(state, action, reward, done, new_state)
            exp_buf.append(exp)
            if done:
                state = env.reset()
                state = np.swapaxes(state, 0 ,2)
                state = np.swapaxes(state,0,1)
                episode_counter += 1
                print("[INFO] episode:{} total reward:{} epsilon:{} loss:{}".format(episode_counter,episode_reward,epsilon,loss))
                total_reward.append(episode_reward)
                episode_reward = 0
                end_t = time.time()
                end_f = frame_id
                print("[INFO] mean_reward:{} speed:{} f/s".format(np.mean(total_reward[-20:]),(end_f - start_f) / (end_t - start_t)))
                start_f = end_f
                start_t = end_t
                
                if episode_counter % SAVE_EPISODE == 0:
                    print("[INFO] MODEL SAVED")
                    saver.save(sess, "./model/weigts")
            else:
                state = new_state
            if len(exp_buf) < REPLAY_START_SIZE:
                continue
            
            if frame_id % SYNC_TARGET_FRAMES == 0:
                num_of_op = qnet.copy_model_parameters(sess, net, tgt_net)
                print("[INFO] copied neural nets {} set of params copied".format(num_of_op))
                
                
            batch = exp_buf.sample(BATCH_SIZE)
            states_, actions_, rewards_, dones_, new_states_ = batch
            labels = calculate_label_qvals(sess,batch,target_net=tgt_net)
            
            loss = net.optimize(sess, states_, actions_, labels)
            
            
            
            
            
                
             
            
        