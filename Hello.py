import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adamax
import keras
import os
from PIL import Image

# Load model with weights
def load_model_with_weights(weights_path):
    img_shape = (224, 224, 3)
    base_model = VGG16(weights='imagenet', input_shape=img_shape, include_top=False, pooling=None)
    base_model.trainable = True
    last_layer = base_model.get_layer('block5_pool')
    last_output = last_layer.output
    x = keras.layers.GlobalMaxPooling2D()(last_output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(2, activation='sigmoid')(x)
    model = tf.keras.Model(base_model.input, x, name="VGG16_model")
    model.load_weights(weights_path)
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Predictions using ensemble model
def ensemble_predict(models, weights, image_array):
    predictions = [model.predict(image_array) * weight for model, weight in zip(models, weights)]
    return tf.reduce_sum(predictions, axis=0)

def run():
    st.title("Leukemia Detection using Ensemble CNN")
    st.write("""
    This application uses an ensemble of Convolutional Neural Networks (CNNs) to detect leukemia from images of blood cell samples.
    Choose an image containing blood cell samples, and the ensemble model will predict whether the image contains cancerous cells.
    """)

    # Load multiple models with different weights
    models = []
    weights = []

    # Load single base model
    base_model_path = 'Weights.h5'
    single_base_model = load_model_with_weights(base_model_path)

    # Assume you have multiple weights files in the directory
    weights_directory = 'EnsembleWeightsDirectory'
    weights_files = os.listdir(weights_directory)
    for weight_file in weights_files:
        weight_path = os.path.join(weights_directory, weight_file)
        model = load_model_with_weights(weight_path)
        models.append(model)
        
        # Equal weights for simplicity
        weights.append(1.0)

    # Callback function to update displayed content after image selection
    def selected_image_callback(selected_image_path):
        if selected_image_path:
            img = Image.open(selected_image_path)
            img = img.resize((224, 224))  # Resize image to match model input size
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch
    
            # Extract image filename to check if it contains "Cancer"
            image_filename = os.path.basename(selected_image_path)
            is_cancer_image = "Cancer" in image_filename
    
            # Initialize lists to store predictions and interpretations for each model
            ensemble_predictions_list = []
            interpretation_list = []
    
            # Run predictions using ensemble model
            for model, weight in zip(models, weights):
                predictions = model.predict(img_array)
                ensemble_predictions_list.append(predictions)
                
                # Interpret predictions
                is_cancerous = predictions[0][1] > 0.5
                confidence = predictions[0][1] if is_cancerous else predictions[0][0]
                interpretation_list.append(f"The model predicts with a confidence of {confidence:.2f} that the image {'contains' if is_cancerous else 'does not contain'} cancerous cells.")
    
            # Congregate ensemble results
            ensemble_results = ensemble_predict(models, weights, img_array)
            is_cancerous_ensemble = ensemble_results[0][1] > 0.5
            confidence_ensemble = ensemble_results[0][1] if is_cancerous_ensemble else ensemble_results[0][0]
            interpretation_list.append(f"The ensemble model predicts that the image {'contains' if is_cancerous_ensemble else 'does not contain'} cancerous cells.")
    
            # Display results
            st.write("### Ensemble Model Predictions:")
            for i, predictions in enumerate(ensemble_predictions_list):
                st.write(f"Model {i+1} Predictions:", interpretation_list[i])
    
            st.write("### Ensemble Model Congregated Results:")
            st.write(f"Predictions:", interpretation_list[-1])
    
            # Run predictions using single base model
            single_model_predictions = single_base_model.predict(img_array)
            is_cancerous_single = single_model_predictions[0][1] > 0.5
            confidence_single = single_model_predictions[0][1] if is_cancerous_single else single_model_predictions[0][0]
            interpretation_single = f"The base model predicts that the image {'contains' if is_cancerous_single else 'does not contain'} cancerous cells."
    
            st.write("### Single Base Model Predictions:")
            st.write("Interpretation:", interpretation_single)
    
            # Check if the image is labeled as "Cancer" in the filename
            if is_cancer_image:
                if is_cancerous_ensemble:
                    st.success("Ensemble model predicts accurately that the image contains cancer.")
                else:
                    st.warning("Ensemble model predicts inaccurately that the image does not contain cancer.")
                st.write("Visit our github page: https://github.com/Anarchist1984/leukemia-ml")
                st.warning(warning)
            else:
                if is_cancerous_ensemble:
                    st.warning("Ensemble model predicts inaccurately that the image contains cancer.")
                else:
                    st.success("Ensemble model predicts accurately that the image does not contain cancer.")
                st.write("Visit our github page: https://github.com/Anarchist1984/leukemia-ml")
                st.warning(warning)
    
        else:
            st.warning("No image selected.")


    # Display the image grid
    image_directory = "StreamlitDirectory"
    display_image_grid(image_directory, selected_image_callback)

def display_image_grid(image_dir, selected_image_callback):

    # Get the list of image files in the specified directory
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if not image_files:
        st.warning("No image files found in the specified directory.")
        return

    # Display images in a single row
    num_images = len(image_files)
    num_columns = min(num_images, 3)  # Display up to 3 images
    selected_image_index = None

    columns = st.columns(num_columns)
    for j in range(num_columns):
        if j < num_images:
            with columns[j]:
                img = Image.open(image_files[j])
                st.image(img, use_column_width=True)  # Remove caption
                
                # Centered button
                button_pressed = st.button(f"Select Image {j+1}", key=f"button_{j}")
                
                if button_pressed:
                    selected_image_index = j

    # Execute callback function with the selected image path
    selected_image_path = image_files[selected_image_index] if selected_image_index is not None else None
    if selected_image_path:
        selected_image_callback(selected_image_path)
    else:
        st.warning("No image selected.")

if __name__ == "__main__":
    run()
