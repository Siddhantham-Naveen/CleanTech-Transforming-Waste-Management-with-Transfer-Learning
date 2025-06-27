from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ===[ 1. Configuration ]===
IMG_SIZE = (96, 96)
BATCH_SIZE = 4
EPOCHS = 10
BASE_DIR = 'dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

print("\n🔧 Starting training setup...")
print(f"📂 Training directory: {TRAIN_DIR}")
print(f"📂 Validation directory: {VAL_DIR}")
print(f"📸 Image size: {IMG_SIZE}, Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}\n")

# ===[ 2. Data Preprocessing ]===
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("\n📚 Detected class labels:", train_data.class_indices)

# ===[ 3. Load Base Model ]===
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# ===[ 4. Add Custom Head ]===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ===[ 5. Compile Model ]===
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("\n✅ Model compiled successfully!")

# ===[ 6. Train Model ]===
print("\n🚀 Starting model training...\n")
history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

# ===[ 7. Save Model ]===
model.save('mobilenetv2_waste.h5')
print("\n💾 Model saved as 'mobilenetv2_waste.h5'\n")
