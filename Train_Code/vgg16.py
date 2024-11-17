import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ตัวแปรสำหรับกำหนด learning rate และจำนวน epochs
learning_rate = 0.001
epochs = 100
best_model_path = 'best_vgg16_model.keras'
batch_size = 32

# ตรวจสอบ GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow กำลังใช้ GPU")
    except RuntimeError as e:
        print(e)
else:
    print("TensorFlow ไม่พบ GPU, กำลังใช้ CPU")

# ข้อมูลสำหรับ training และ validation
train_dir = r'D:\MobileNetV2_Hand_Writing\TrainingData'
val_dir = r'D:\MobileNetV2_Hand_Writing\ValidationData'

# การเตรียมข้อมูลโดยใช้ ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจาก directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# แช่การเรียนรู้ของ base model
base_model.trainable = False

# สร้างโมเดลใหม่โดยต่อชั้นใหม่เข้าไป
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # เพิ่ม L2 regularization
    tf.keras.layers.BatchNormalization(),  # เพิ่ม Batch Normalization
    tf.keras.layers.Dropout(0.5),          # เพิ่ม Dropout
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks สำหรับการหยุดและบันทึกโมเดลที่ดีที่สุด
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

# เทรนโมเดล
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# แสดงกราฟ accuracy และ loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()