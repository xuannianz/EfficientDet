import tensorflow as tf

def tfloader(loader, batch_size, cache_filename=None, validate=False):
    # index to each path sample
    indexes = [i for i in range(len(loader))]
    indexes = tf.random.shuffle(indexes)
    
    # generator to yield data
    def generator(i):
        x, y = loader[i]
        yield tf.squeeze(x), (tf.squeeze(y[0]), tf.squeeze(y[1]))
        
    # take one value to get shape and dtype
    x,y = next(generator(0))
    
    # Tensorflow Dataset API options
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset
        #dataset = dataset.from_tensor_slices(indexes)
        #dataset = dataset.shuffle(2048)
        dataset = dataset.interleave(
                        dataset.from_tensor_slices(loader),
                        cycle_length=1,
                        block_length=1,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        if not validate:
            dataset = dataset.repeat(10)
        
        if cache_filename is not None:
            dataset = dataset.cache(filename=cache_filename)
        
        dataset = dataset.repeat()
    
    return dataset