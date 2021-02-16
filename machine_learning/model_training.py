from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as K
from parmec_analysis.utils import folder_validation, experiment_iteration, save_results
from machine_learning.experiment_summary import summary
import os

NUMCORES = int(os.getenv("NSLOTS", 1))

#sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
#                                        allow_soft_placement=True,
#                                        device_count={'CPU': NUMCORES}))

# Set the Keras TF session
# K.set_session(sess)

config=tf.ConfigProto(inter_op_parallelism_threads=NUMCORES,
   intra_op_parallelism_threads=NUMCORES)


# rlrop = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, cooldown=1)


def run_experiment(experiment):
    trial_name = experiment.trial.trial_name

    folder_validation(trial_name)

    experiment_folder = trial_name + "/" + experiment.name
    folder_validation(experiment_folder)

    summary(experiment)

    exp_i = experiment_iteration(experiment.name, trial_name)

    loss = experiment.trial.loss_function

    experiment.callbacks[0] = experiment.callbacks[0](loss, trial_name, experiment, exp_i,
                                                      experiment.trial.plot_every_n_epochs)

    ###
    
    if len(experiment.callbacks) > 1:
      
      checkpoint_filepath = experiment.name + "_" + str(exp_i) + ".hdf5"
  
      experiment.callbacks[1] = experiment.callbacks[1](
          filepath=checkpoint_filepath,
          save_weights_only=False,
          monitor='val_loss',
          mode='min',
          save_best_only=True)

    ##########################################################

    model_i = experiment.model

    model_i.compile(loss=loss.__name__, optimizer=experiment.trial.optimiser)

    model_fit = model_i.fit(experiment.features.training_set(), experiment.labels.training_set(),
                            validation_data=(experiment.features.validation_set(),
                                             experiment.labels.validation_set()),
                            epochs=experiment.trial.epochs, verbose=2, 
                            callbacks=experiment.callbacks)

    # This section now sorts these lists and passes the lowest to the save file
    loss = model_fit.history['loss']
    loss.sort(reverse=True)

    val_loss = model_fit.history['val_loss']
    val_loss.sort(reverse=True)

    losses = [model_fit.history['loss'][-1], model_fit.history['val_loss'][-1]]

    save_results(experiment.name, trial_name, exp_i, losses)
