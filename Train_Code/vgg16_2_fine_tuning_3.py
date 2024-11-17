import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ตัวแปรสำหรับกำหนด learning rate และจำนวน epochs
learning_rate = 0.00001  # ใช้ learning rate ที่สูงขึ้น
epochs = 100  # เพิ่มจำนวน epochs
best_model_path = 'best_fine_tuned_3_vgg16_2_model.keras'
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
    rotation_range=25,  # ลด rotation range ลงเล็กน้อย
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,  # เพิ่มการซูมเล็กน้อยเพื่อความหลากหลาย
    brightness_range=[0.7, 1.3],  # ปรับช่วงความสว่างเพิ่ม
    horizontal_flip=True,
    fill_mode='nearest'
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

# โหลด VGG16 โมเดล
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# เปิดการเรียนรู้ของ base model
base_model.trainable = True

# กำหนดชั้นที่ต้องการให้ trainable
for layer in base_model.layers[:-8]:  # ปรับให้ layer สุดท้าย 4 ชั้นเป็น trainable
    layer.trainable = False

# สร้างโมเดลใหม่โดยต่อชั้นใหม่เข้าไป
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # ทำ global pooling
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.002))(x)  # L2 regularization
x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization
x = tf.keras.layers.Dropout(0.6)(x)  # ลด Dropout Rate
outputs = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)  # สร้างโมเดลใหม่

# โหลดน้ำหนักจากโมเดลที่ถูกฝึกก่อนหน้านี้
try:
    model.load_weights('best_fine_tuned_2_vgg16_2_model.keras')
    print("โหลดน้ำหนักโมเดลที่ถูกฝึกไว้แล้ว")
except Exception as e:
    print(f"ไม่สามารถโหลดน้ำหนักโมเดลได้: {e}")

# คอมไพล์โมเดล
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks สำหรับการหยุดและบันทึกโมเดลที่ดีที่สุด
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)  # เปลี่ยนเป็น monitor val_accuracy
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1)

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
