import tensorflow as tf
from get_data import  data
import numpy as np
from sklearn.model_selection import  train_test_split

def tf_nn(x):

    W1 = tf.Variable(tf.random_normal([300,128],stddev = 0.03), name = 'W1')
    b1 = tf.Variable(tf.random_normal([128]),name = 'b1')

    W2 = tf.Variable(tf.random_normal([128,64],stddev = 0.03), name = 'W2')
    b2 = tf.Variable(tf.random_normal([64]), name='b2')

    W3 = tf.Variable(tf.random_normal([64, 10], stddev=0.03), name='W3')
    b3 = tf.Variable(tf.random_normal([10]), name='b3')

    W4 = tf.Variable(tf.random_normal([10, 1], stddev=0.03), name='W4')
    b4 = tf.Variable(tf.random_normal([2]), name='b4')

    hidden_l1_out = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
    hidden_l2_out = tf.nn.relu(tf.add(tf.matmul(hidden_l1_out,W2),b2))
    hidden_l3_out = tf.nn.relu(tf.add(tf.matmul(hidden_l2_out,W3),b3))

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_l3_out,W4),b4))

    return y_

if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 50
    batch_size = 128
    X_data = np.loadtxt('train.txt', dtype=float)
    y_data = np.loadtxt('test.txt', dtype = float)
    X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2,random_state=123)
    
    X = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, [None, 1])

    y_ = tf_nn(X)

    root_square = tf.reduce_sum(tf.square(y_ - y))
    optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(root_square)

    init_op = tf.global_variables_initializer()
    correct_prediction = tf.equal(y, tf.round(y_))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(X_train) / batch_size)
        if total_batch * batch_size > len(X_train):
            total_batch += 1
        for epoch in range(epochs):
            avg_cost = 0
            begin_batch = 0
            end_batch = begin_batch + batch_size
            for i in range(total_batch):
                if ((i+1) * batch_size) > len(X_train):
                    batch_x = X_train[i*batch_size,len(X_train) - (i*batch_size) -1,:]
                    batch_y = y_train[i * batch_size, len(X_train) - (i * batch_size) - 1]
                batch_x = X_train[begin_batch:end_batch,:]
                batch_y = y_train[begin_batch:end_batch].reshape((128,1))

                begin_batch = end_batch + 1
                end_batch = begin_batch + batch_size
                _, c = sess.run([optimiser, root_square],
                                feed_dict={X: batch_x, y: batch_y})
            avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

        print(sess.run(accuracy, feed_dict={X: X_test, y: y_test[:,np.newaxis]}))
