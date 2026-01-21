import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from config.config import CFG, IS_GPU

def build_model(num_classes, initial_freeze=True):
    """
    Builds the ResNet50V2 (or MobileNetV2) model with a custom head.
    """
    tf.keras.backend.clear_session()
    
    BACKBONE = ResNet50V2 if IS_GPU else MobileNetV2
    BACKBONE_NAME = 'ResNet50V2' if IS_GPU else 'MobileNetV2'

    inputs = Input(shape=(*CFG.IMG_SIZE, 3))

    try:
        base = BACKBONE(include_top=False, weights='imagenet', input_tensor=inputs)
        print(f"Loaded {BACKBONE_NAME} with ImageNet weights")
    except Exception as e:
        print(f"Failed loading ImageNet weights for {BACKBONE_NAME}: {e}")
        base = BACKBONE(include_top=False, weights=None, input_tensor=inputs)
        print(f"{BACKBONE_NAME} instantiated without pretrained weights")

    if initial_freeze:
        base.trainable = False
        print(f"Base model ({BACKBONE_NAME}) is initially frozen.")
    else:
        # This is where the model is configured to be fully trainable
        base.trainable = True
        print(f"Base model ({BACKBONE_NAME}) is fully trainable for fine-tuning.")

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x) 

    model = Model(inputs, outputs)
    
    current_lr = CFG.LR if initial_freeze else CFG.FINE_TUNE_LR
    model.compile(
        optimizer=Adam(learning_rate=current_lr, amsgrad=True),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
