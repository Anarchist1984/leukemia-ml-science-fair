import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adamax
import keras
import os
from PIL import Image

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

def ensemble_predict(models, weights, image_array):
    predictions = [model.predict(image_array) * weight for model, weight in zip(models, weights)]
    return tf.reduce_sum(predictions, axis=0)

def run():
    st.set_page_config(
        page_title="LeukemiaCNN",
        page_icon="👋",
    )

    image_directory = 'StreamlitDirectory'

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
        
        # You can assign different weights to each model
        # For simplicity, let's assume equal weights for now
        weights.append(1.0)

    # Callback function to update the displayed content after image selection
    def selected_image_callback(selected_image_path):
        if selected_image_path:
            img = Image.open(selected_image_path)
            img = img.resize((224, 224))  # Resize image to match model input size
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            # Initialize lists to store predictions and interpretations for each model
            ensemble_predictions_list = []
            single_model_predictions_list = []
            interpretation_list = []

            # Run predictions using ensemble model
            for model, weight in zip(models, weights):
                predictions = model.predict(img_array)
                ensemble_predictions_list.append(predictions)
                
                # Interpret predictions
                is_cancerous = predictions[0][1] > 0.5
                single_model_predictions_list.append(predictions)
                if is_cancerous:
                    interpretation_list.append("The model predicts that the image contains cancerous cells.")
                else:
                    interpretation_list.append("The model predicts that the image does not contain cancerous cells.")

            # Run predictions using single base model
            single_model_predictions = single_base_model.predict(img_array)
            single_model_predictions_list.append(single_model_predictions)

            # Compare predictions
            # Do whatever comparison you want here
            # For simplicity, let's just print the predictions and interpretations for now
            st.write("### Ensemble Model Predictions:")
            for i, predictions in enumerate(ensemble_predictions_list):
                st.write(f"Model {i+1} Predictions:", predictions)
                st.write(f"Interpretation:", interpretation_list[i])

            st.write("### Single Base Model Predictions:")
            st.write(single_model_predictions_list)

        else:
            st.warning("No image selected.")

    # Display the image grid
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
                st.image(img, caption=f"Image {j+1}", use_column_width=True)
                
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