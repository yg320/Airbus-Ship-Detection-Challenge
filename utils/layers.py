from tensorpack import layer_register
import tensorflow as tf

@layer_register(log_shape=False, use_scope=False)
def CropAndConcat(dec_layer, enc_layer):

    crop_dec = 2
    new_dec_layer = dec_layer[:, crop_dec: -crop_dec, crop_dec: -crop_dec, :]

    crop_enc = int((enc_layer.shape[1] - new_dec_layer.shape[1]) // 2)
    new_enc_layer = enc_layer[:, crop_enc: -crop_enc, crop_enc: -crop_enc, :]
    print(new_enc_layer)
    print(new_dec_layer)
    return tf.concat([new_enc_layer, new_dec_layer], 3)
