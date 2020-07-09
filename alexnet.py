import tensorflow as tf
import os

#以类的形式建立AlexNet
class AlexNet(object):
    #初始化参数，keep_prob——Dropout中保留隐藏层单元的概率    input_width、input_height——输入图像的参数
    #          num_classes——多分类问题的分类种类     batch_size——batch的大小
    def __init__(self, keep_prob, input_heigth, input_width, num_classes, batch_size):
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.num_classes = num_classes

        #处理输入图像过小的问题
        if (input_width + input_heigth)/2 < 128:
            self.padding = 'same'
        else:
            self.padding = 'valid'

        #加载模型或者创建模型
        if os.path.exists('./model/AlexNet'):
            self.model = tf.keras.models.load_model('./model/AlexNet')
            print('AlexNet Model has been loaded...')
        else:
            self.build_model(input_heigth, input_width)

    #AlexNet模型，以函数API形式构建
    def build_model(self, input_heigth, input_width):
        tf.keras.backend.clear_session()
        inputs = tf.keras.layers.Input(shape=[input_heigth, input_width, 3])

        #layer 1
        conv1_1 = tf.keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.zeros_initializer())(inputs)
        conv1_2 = tf.keras.layers.Conv2D(filters=48, kernel_size=11, strides=4, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.zeros_initializer())(inputs)

        #layer 2
        conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.ones_initializer())(conv1_1)
        conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.ones_initializer())(conv1_2)

        conv2_1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding=self.padding)(conv2_1)
        conv2_2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding=self.padding)(conv2_2)

        conv2_1 = tf.keras.layers.BatchNormalization()(conv2_1)
        conv2_2 = tf.keras.layers.BatchNormalization()(conv2_2)

        conv2 = tf.keras.layers.Concatenate()([conv2_1, conv2_2])

        #layer 3
        conv3_1 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.zeros_initializer())(conv2)
        conv3_2 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.zeros_initializer())(conv2)

        conv3_1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding=self.padding)(conv3_1)
        conv3_2 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding=self.padding)(conv3_2)

        conv3_1 = tf.keras.layers.BatchNormalization()(conv3_1)
        conv3_2 = tf.keras.layers.BatchNormalization()(conv3_2)

        #layer 4
        conv4_1 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.ones_initializer())(conv3_1)
        conv4_2 = tf.keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.ones_initializer())(conv3_2)

        #layer 5
        conv5_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.zeros_initializer())(conv4_1)
        conv5_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding=self.padding,
                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                         bias_initializer=tf.zeros_initializer())(conv4_2)
        conv5 = tf.keras.layers.Concatenate()([conv5_1, conv5_2])

        #layer 6
        flatten = tf.keras.layers.Flatten()(conv5)
        dense1 = tf.keras.layers.Dense(1024, activation='relu',
                                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                       bias_initializer=tf.ones_initializer())(flatten)
        dense1 = tf.keras.layers.Dropout(rate=self.keep_prob)(dense1)

        #layer 7
        dense2 = tf.keras.layers.Dense(128, activation='relu',
                                       kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                       bias_initializer=tf.ones_initializer())(dense1)
        dense2 = tf.keras.layers.Dropout(rate=self.keep_prob)(dense2)

        #layer 8
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(dense2)

        #整合模型
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        #设置优化器、损失函数、评估标准
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        #损失函数和评估标准均要求labels为one_hot形式
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        metric = tf.keras.metrics.CategoricalAccuracy()

        #模型编译
        self.model.compile(optimizer, loss_func, [metric])

    #使AlexNet类可以调用，请确保数据形状为(None, input_height, input_width, 3)
    #                        标签形状为(None, )
    #__call__函数会将标签整理为one_hot形式
    def __call__(self, train_features, train_labels, test_features, test_labels, epochs):
        train_features = tf.constant(train_features, dtype=tf.float32)
        train_labels = tf.constant(tf.one_hot(train_labels, self.num_classes), dtype=tf.float32)

        test_features = tf.constant(test_features, dtype=tf.float32)
        test_labels = tf.constant(tf.one_hot(test_labels, self.num_classes), dtype=tf.float32)

        ds_train = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).shuffle(buffer_size=100).batch(batch_size=self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
        ds_test = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).shuffle(buffer_size=100).batch(batch_size=self.batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

        self.model.fit(ds_train, validation_data=ds_test, epochs=epochs)

        if not os.path.exists('./model/AlexNet'):
            os.makedirs('./model/Alexnet')

        self.model.save('./model/AlexNet', save_format='tf')
        print('The AlexNet has been saved...')
