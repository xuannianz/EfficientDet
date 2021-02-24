''' TODO '''

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def serialize_example(image, regression, class_):
    feature = {
      'image': _bytes_feature(image),
      'reg': _bytes_feature(regression),
      'clas': _bytes_feature(class_),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecords(tfrecords_path, mode, generator):
    
    img, (reg, clas) = generator[0]
    
    # get shapes without batch size and save
    img_shape = np.squeeze(img).shape[1:]
    reg_shape = np.squeeze(reg).shape[1:]
    class_shape = np.squeeze(clas).shape[1:]
    
    np.savez(os.path.join(tfrecords_path, mode+'_shapes.npz'),
            image=img_shape, reg=reg_shape, clas=class_shape)
    
    # create and save tfrecords
    it = 10 if mode=='train' else 1
    
    with tf.io.TFRecordWriter(os.path.join(tfrecords_path, mode+'.tfrec'), options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        for i in range(it):
            print(f'Iteration {i+1}/{it}...')
            for img, (reg, clas) in tqdm(generator):
                img = np.squeeze(img)
                reg = np.squeeze(reg)
                clas = np.squeeze(clas)

                # iterate over batch
                for i in range(img.shape[0]):
                    example = serialize_example(img[i], reg[i], clas[i])
                    writer.write(example)
            
def get_loader(tfrecords_path, mode, batch_size):
    
    shapes = np.load(os.path.join(tfrecords_path, mode+'_shapes.npz'))
    def decode(x, shape):
        x = tf.io.decode_raw(x, tf.float32)
        return tf.reshape(x, shape)
    
    def read_data(example):
        LABELED_TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'reg': tf.io.FixedLenFeature([], tf.string),
            'clas': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
        image = decode(example['image'], shapes['image'])
        reg   = decode(example['reg'], shapes['reg'])
        clas  = decode(example['clas'], shapes['clas'])
        return image, (reg, clas)
    
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(os.path.join(tfrecords_path, mode+'.tfrec'),\
                                      compression_type='GZIP',\
                                      num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_data)
    dataset = dataset.batch(batch_size)
    #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    
    return dataset