from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config import CFG

def make_generators(batch_size):
    """
    Creates training and validation data generators.
    """
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    print(f"Creating generators from {CFG.TRAIN_DIR} and {CFG.TEST_DIR}")

    train_gen = train_aug.flow_from_directory(
        CFG.TRAIN_DIR,
        target_size=CFG.IMG_SIZE,
        batch_size=batch_size, 
        class_mode='categorical',
        shuffle=True,
        seed=CFG.SEED
    )
    val_gen = val_aug.flow_from_directory(
        CFG.TEST_DIR,
        target_size=CFG.IMG_SIZE,
        batch_size=batch_size, 
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen
