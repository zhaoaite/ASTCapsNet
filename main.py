import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from capsNet import CapsNet
from sklearn import metrics  
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')




def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)

# Random Forest Classifier  
def random_forest_classifier(train_x, train_y):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=10)  
    model.fit(train_x, train_y) 
    return model  
  

def train(model, supervisor, num_label):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step
                
                
                if global_step % cfg.train_sum_freq == 0:
                    argmax_idx,_,_,_,_,caps2, loss, summary_str, dense1_index, dense2_index, labels = sess.run([model.argmax_idx,model.train_op,model.train_c1,model.train_c2,model.train_c3, model.caps2,model.total_loss
                                                , model.train_summary, model.dense1_index, model.dense2_index, model.labels]) 
#                    argmax_idx,_,_,loss, summary_str, lstm_index, labels = sess.run([model.argmax_idx,model.train_op,model.train_c1, model.total_loss
#                                                , model.train_summary, model.lstm_index, model.labels]) 
                    print(caps2)
#                    max_index_list= np.ones([cfg.batch_size, ])
                    index_list=np.c_[dense1_index,dense2_index,argmax_idx]
#                    index_list=np.c_[lstm_index,argmax_idx]
                    #Create a Gaussian Classifier
                    bayes_model = GaussianNB()
                    # Train the model using the training sets 
                    bayes_model.fit(index_list, labels)
                    #Predict Output 
                    bayes_predicted= bayes_model.predict(index_list)
                    train_acc=np.sum(np.equal(bayes_predicted,labels).astype(int))
                    
##                    The voting process
#                    for i in range(cfg.batch_size):
#                        max_index=np.argmax(np.bincount(index_list[i]))
#                        max_index_list[i]=max_index
#                        
#                    train_acc=np.sum(np.equal(max_index_list,labels).astype(int))
  
#                    RF process                    
#                    rf_model=random_forest_classifier(rf_input, labels)
#                    print(rf_model.feature_importances_)
#                    for line in rf_input:
#                        max_impor=np.argmax(np.bincount(line))
#    #                   RF training 
#                    _,rf_loss_op,rf_accuracy_op = sess.run([model.rf_train_op,model.rf_loss_op,model.rf_accuracy_op])   
#                    print('RF:',str(rf_accuracy_op))
                    
                    
##                    random forest testing
#                    rf_model_conv1=random_forest_classifier(np.reshape(conv1,(cfg.batch_size,-1)), labels)
#                    predict_conv1 = rf_model_conv1.predict(np.reshape(conv1,(cfg.batch_size,-1)))
#                    print(rf_model_conv1.feature_importances_)
#                    rf_model_caps1=random_forest_classifier(np.reshape(caps1,(cfg.batch_size,-1)), labels)
#                    predict_caps1 = rf_model_caps1.predict(np.reshape(caps1,(cfg.batch_size,-1)))
#                    print(predict_caps1)
#                    rf_model_caps2=random_forest_classifier(np.reshape(caps2,(cfg.batch_size,-1)), labels)
#                    predict_caps2 = rf_model_caps2.predict(np.reshape(caps2,(cfg.batch_size,-1)))
#                    print(predict_caps2)
#                    print(labels)
#                    print(argmax_idx)
                    
                    #caps2=np.reshape(caps2,(cfg.batch_size,16*4))
                    #print(caps2.shape)
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        argmax_idx, dense1_index, dense2_index, labels = sess.run([model.argmax_idx, model.dense1_index, model.dense2_index, model.labels],
                                                                                  {model.X: valX[start:end], model.labels: valY[start:end]}) 

#                        argmax_idx, lstm_index, labels = sess.run([model.argmax_idx, model.lstm_index, model.labels],
#                                                                                  {model.X: valX[start:end], model.labels: valY[start:end]}) 

#                        index_list=np.c_[lstm_index,argmax_idx]
#                        max_index_list= np.ones([cfg.batch_size, ])
                        index_list=np.c_[dense1_index,dense2_index,argmax_idx]
                        
                        #Create a Gaussian Classifier
                        bayes_model = GaussianNB()
                        # Train the model using the training sets 
                        bayes_model.fit(index_list, labels)
                        #Predict Output 
                        bayes_predicted= bayes_model.predict(index_list)
                        acc=np.sum(np.equal(bayes_predicted,labels).astype(int))
                        
#                        for i in range(cfg.batch_size):
#                            max_index=np.argmax(np.bincount(index_list[i]))
#                            max_index_list[i]=max_index
#                            
#                        acc=np.sum(np.equal(max_index_list,labels).astype(int))
                        
#                        acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()



#extract the spacial feature
def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')
        #save to txt
        caps2 = sess.run(model.caps2)
#        caps2,decoded = sess.run([model.caps2,model.decoded])
        caps2=np.reshape(caps2,(cfg.batch_size,16*model.n_classes))
        test_acc = 0
        predict_lists=np.array([])
        truelabel_lists=np.array([])
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            argmax_idx, dense1_index, dense2_index, labels = sess.run([model.argmax_idx, model.dense1_index, model.dense2_index, model.labels],
                                                                      {model.X: teX[start:end], model.labels: teY[start:end]}) 
#            argmax_idx, lstm_index, labels = sess.run([model.argmax_idx, model.lstm_index, model.labels],
#                                                                                  {model.X: teX[start:end], model.labels: teY[start:end]}) 

#            max_index_list= np.ones([cfg.batch_size, ])
            index_list=np.c_[dense1_index,dense2_index,argmax_idx]
#            for i in range(cfg.batch_size):
#                max_index=np.argmax(np.bincount(index_list[i]))
#                max_index_list[i]=max_index
                
#            acc=np.sum(np.equal(max_index_list,labels).astype(int))
            
            
#            index_list=np.c_[lstm_index,argmax_idx]
            #Create a Gaussian Classifier
            bayes_model = GaussianNB()
            # Train the model using the training sets 
            bayes_model.fit(index_list, labels)
            #Predict Output 
            bayes_predicted= bayes_model.predict(index_list)
            predict_lists=np.r_[predict_lists,bayes_predicted] if predict_lists.size else bayes_predicted
            truelabel_lists=np.r_[truelabel_lists,labels] if truelabel_lists.size else labels
            acc=np.sum(np.equal(bayes_predicted,labels).astype(int))
#            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')
        np.savetxt('results/predict_lists.csv',predict_lists)
        np.savetxt('results/truelabel_lists.csv',truelabel_lists)

def main(_):
    
    #remove cpu occupation
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    tf.reset_default_graph()
    tf.logging.info(' Loading Graph...')
    num_label = 52
    # reset graph
#    tf.reset_default_graph()
    model = CapsNet()
    tf.logging.info(' Graph loaded')
#    sv= tf.train.MonitoredTrainingSession(model.graph, checkpoint_dir=cfg.logdir,save_checkpoint_secs=3)
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir,save_model_secs=0)
    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model, sv, num_label)
        tf.logging.info('Training done')
    else:
        evaluation(model, sv, num_label)

if __name__ == "__main__":
    tf.app.run()
