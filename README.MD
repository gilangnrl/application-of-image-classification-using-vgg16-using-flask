# Image Classification using VGG16

This repository contains a machine learning project for image classification utilizing the VGG16 model.

## Dataset

The project uses two datasets:
- Rock Paper Scissors
- Corel 10k

## Dependencies

To run this project, make sure you have the following dependencies installed:
- opencv-python
- numpy
- flask
- tensorflow
- livereload
- pillow

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

Follow these steps to use the project:
1. Clone this repository:
   ```bash
   git clone https://github.com/gilangnrl/application-of-image-classification-using-vgg16-using-flask
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   flask run
   ```

## Project Structure

- `model/`: Contains model used with `.h5`,`hdf5` format (folder empty because the size is to large).
- `app.py`: Flask application for serving the model and routing.
- `templates/`: Contains HTML templates for the web interface.
- `static/`: Directory for static files (CSS, images, etc.).
- `requirements.txt`: File containing all required Python packages and versions.

Feel free to explore and modify the code as needed!
```

Replace `"your-username"` and `"your-repo"` with your GitHub username and the repository name, respectively. Customize the description and details as necessary for your project. This README.md provides an overview of the project, its usage instructions, dependencies, and a brief project structure explanation.
