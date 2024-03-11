# Leukemia Detection using Ensemble CNN

This is a Streamlit web application that uses an ensemble of Convolutional Neural Networks (CNNs) to detect leukemia from images of blood cell samples. The application allows users to upload an image containing blood cell samples, and the ensemble model will predict whether the image contains cancerous cells.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Anarchist1984/leukemia-ml.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. The web application will open in your default web browser. You can then select an image containing blood cell samples to get predictions from the ensemble model.

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Streamlit
- PIL (Python Imaging Library)

## File Structure

- `Hello.py`: Main Streamlit application file.
- `requirements.txt`: File containing Python dependencies.
- `EnsembleWeightsDirectory/`: Directory containing weights files for ensemble models.
- `StreamlitDirectory/`: Directory containing image files for testing the application.
- `Weights.h5`: Weight file for the single base model.
- `README.md`: Documentation file (you are here).

## Credits

- This project uses the VGG16 model pre-trained on ImageNet for feature extraction.
- The ensemble model combines predictions from multiple VGG16-based models.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the Apache license, but anyone feel free to use it.
