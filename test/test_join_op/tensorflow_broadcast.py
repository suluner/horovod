import tensorflow as tf

import horovod.tensorflow as hvd

tf.enable_eager_execution()

def test_broadcast_join(root_rank=0):
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    broadcast_num = 4 if rank == 0 else 2

    for i in range(broadcast_num):
        tensor = tf.ones([17] * 2) * rank
        root_tensor = tf.ones([17] * 2) * root_rank
        broadcasted_tensor = hvd.broadcast(tensor, root_rank)

if __name__ == '__main__':
    test_broadcast_join()
    hvd.join()
