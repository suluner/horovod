import tensorflow as tf
import horovod.tensorflow as hvd
import os


tf.logging.set_verbosity(tf.logging.INFO)

train_file = "/mnt/dl-storage/dg-glusterfs-1/public/boston_housing_data/boston_train-*.csv"
test_file = "/mnt/dl-storage/dg-glusterfs-1/public/boston_housing_data/boston_test.csv"
model_dir = os.environ['PERSISTENT_DIR']
train_batch_size = 10
test_batch_size = 100

feature_names = [
    "crim", "zn", "indus", "nox", "rm", "age", "dis", "tax", "ptratio"
]

def decode_csv(line):
    parsed_line = tf.decode_csv(
        line, [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])
    label = parsed_line[-1:]  # Last element is the label
    del parsed_line[-1]  # Delete last element
    features = parsed_line  # Everything (but last element) are the features
    d = dict(zip(feature_names, features)), label
    return d


def train_input_fn(file_path, perform_shuffle=True, repeat_count=None):

    batch_size = train_batch_size
    dataset = tf.data.Dataset.list_files(file_path, shuffle=False)
    # Different process has different number of samples,
    dataset = dataset.shard(hvd.size(), hvd.rank())
    dataset = dataset.interleave(tf.data.TextLineDataset, cycle_length=1).map(decode_csv)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def eval_input_fn(file_path, perform_shuffle=False, repeat_count=1):

    dataset = (
        tf.data.TextLineDataset(file_path)  # Read text file
        .skip(1)
        .map(decode_csv))  # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(test_batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


hvd.init()
model_dir = model_dir if hvd.rank() == 0 else None
bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
# Create the feature_columns, which specifies the input to our model.
# All our input features are numeric, so use numeric_column for each one.
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
cfg = tf.estimator.RunConfig(session_config=config,
                             save_summary_steps=10,
                             log_step_count_steps=1)

optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
optimizer = hvd.DistributedOptimizer(optimizer)
regressor = tf.estimator.DNNRegressor(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    model_dir=model_dir,
    config=cfg,
    optimizer=optimizer)

regressor.train(
    input_fn=lambda: train_input_fn(train_file, True, 100),
    hooks=[bcast_hook])


# Evaluate loss over one epoch of test_set.
if hvd.rank() == 0:
    ev = regressor.evaluate(input_fn=lambda: eval_input_fn(test_file, False, 1))
    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

# Wait for other process to end
hvd.join()
