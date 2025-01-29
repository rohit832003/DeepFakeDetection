import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Flask Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Enhanced ELA Processing
def apply_ela(image_path, quality=90):
    temp_path = "temp.jpg"
    try:
        original = Image.open(image_path).convert("RGB")
        original.save(temp_path, "JPEG", quality=quality)
        compressed = Image.open(temp_path)
        
        ela_image = np.array(original).astype(np.int16) - np.array(compressed).astype(np.int16)
        ela_image = (15 * np.abs(ela_image)).clip(0, 255).astype(np.uint8)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return ela_image
    except Exception as e:
        print(f"ELA Error: {str(e)}")
        return None

# Model Loading with EfficientNet Preprocessing
def load_model():
    model = tf.keras.models.load_model('deepfake_model.h5', compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return model

# Unified Prediction Function
def predict_media(file_path, media_type):
    model = load_model()
    
    if media_type == 'image':
        ela_image = apply_ela(file_path)
        if ela_image is None:
            return "Error processing image"
        
        processed = cv2.resize(ela_image, (224, 224))
        processed = preprocess_input(processed)  # EfficientNet preprocessing
        prediction = model.predict(np.expand_dims(processed, axis=0))[0][0]
        
        return {
            'type': 'image',
            'prediction': 'Fake' if prediction > 0.5 else 'Real',
            'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
        }
    
    elif media_type == 'video':
        cap = cv2.VideoCapture(file_path)
        predictions = []
        
        while cap.isOpened() and len(predictions) < 100:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)
            ela_frame = apply_ela(frame_path)
            
            if ela_frame is not None:
                processed = cv2.resize(ela_frame, (224, 224))
                processed = preprocess_input(processed)
                pred = model.predict(np.expand_dims(processed, axis=0))[0][0]
                predictions.append(pred)
            
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        cap.release()
        if not predictions:
            return "No valid frames processed"
        
        avg_pred = np.mean(predictions)
        return {
            'type': 'video',
            'prediction': 'Fake' if avg_pred > 0.5 else 'Real',
            'confidence': float(avg_pred if avg_pred > 0.5 else 1 - avg_pred),
            'frames_analyzed': len(predictions)
        }

# Flask Routes
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            try:
                media_type = 'image' if filename.lower().split('.')[-1] in {'png', 'jpg', 'jpeg'} else 'video'
                result = predict_media(save_path, media_type)
                return render_template('result.html', result=result, filename=filename)
            except Exception as e:
                return f"Error: {str(e)}"
            finally:
                if os.path.exists(save_path):
                    os.remove(save_path)
    
    return render_template('upload.html')

# Training Function with Correct Preprocessing
def train_model():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,  # EfficientNet preprocessing
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'dataset/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'dataset/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    base_model = tf.keras.applications.EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers[:150]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[
            callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            callbacks.ModelCheckpoint('deepfake_model.h5', save_best_only=True)
        ]
    )

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_model()
    else:
        app.run(host='0.0.0.0', port=5000, debug=True)