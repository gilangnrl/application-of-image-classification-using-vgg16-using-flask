# Image Classification using VGG16
![Example](/static/example.jpg)
![Example2](/static/example2.jpg)
This repository contains a machine learning project for image classification utilizing the VGG16 model.

## Dataset

The project uses two datasets:
- Rock Paper Scissors, and
- Corel 10k

## Dependencies

To run this project, make sure you have the following dependencies installed:
- opencv-python
- numpy
- flask
- tensorflow
- livereload
- pillow

## Usage

Follow these steps to use the project:
1. Clone this repository:
   ```bash
   git clone https://github.com/gilangnrl/application-of-image-classification-using-vgg16-using-flask
   ```
2. Navigate to the project directory:
   ```bash
   cd application-of-image-classification-using-vgg16-using-flask
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the model [Here](https://drive.google.com/drive/folders/1dIfktYxRH76ktCSWUakhoAb07rppCGgw?usp=sharing), and copy into `model/`.
5. Run the Flask application:
   ```bash
   flask run
   ```

## Project Structure

- `model/`: Contains model used with `.h5` format (folder empty because the size is too large. You can download the model [Here](https://drive.google.com/drive/folders/1dIfktYxRH76ktCSWUakhoAb07rppCGgw?usp=sharing)).
- `app.py`: Flask application for serving the model and routing.
- `templates/`: Contains HTML templates for the web interface.
- `static/`: Directory for static files (CSS, images, etc.).
- `requirements.txt`: File containing all required Python packages and versions.

Feel free to explore and modify the code as needed!
```
