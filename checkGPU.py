import tensorflow as tf

# 確認是否有 GPU 被 TensorFlow 探測到
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))