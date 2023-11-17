from flask import Flask, jsonify, Response, g, request
import config
import json
import time
from PIL import Image
import io
import easyocr
import uuid, os , re
from blueprints.activities import activities
from flask_cors import CORS
import numpy as np
import cv2

SECRET_API_KEY = 'secretkey'
# Create an EasyOCR reader
reader = easyocr.Reader(['en'])

def create_app():
  app = Flask(__name__)
  app.register_blueprint(activities, url_prefix="/api/v1/activities")
  CORS(app, origins=["http://localhost:8000", "https://wasc-ui-hhbackhwia-el.a.run.app"])


  # Error 404 handler
  @app.errorhandler(404)
  def resource_not_found(e):
    return jsonify(error=str(e)), 404
  # Error 405 handler
  @app.errorhandler(405)
  def resource_not_found(e):
    return jsonify(error=str(e)), 405
  # Error 401 handler
  @app.errorhandler(401)
  def custom_401(error):
    return Response("API Key required.", 401)
  
  #Endpoint: version 
  @app.route("/version", methods=["GET"], strict_slashes=False)
  def version():
    response_body = {
        "success": 1,
    }
    return jsonify(response_body)
  
  #Endpoint: ping
  @app.route("/ping")
  def hello_world():
     return "pong"
  
  #Endpoint: ping
  @app.route("/ping/v2")
  def hello_world_v2():
     return "pong v2"

  #Endpoint: predict
  @app.route('/predict', methods=['POST'])
  def predict():

      api_key = request.form['api_key']

      # Verify the API key
      if api_key != SECRET_API_KEY:
          return jsonify({"error": "Invalid API key"}), 401
          
      # Check if the request contains a file
      if 'file' not in request.files:
          return jsonify({"error": "No file provided"}), 400

      file = request.files['file']

      # Get the current script directory
      current_directory = os.path.dirname(os.path.abspath(__file__))

      # Define the subdirectory (TMP_STORAGE_PATH)
      subdirectory = "TMP_STORAGE_PATH"

      # Create the full path for the filename by joining the current directory, subdirectory, and the unique identifier with the file extension
      filename = os.path.join(current_directory, subdirectory, str(uuid.uuid4()) + '.png')

      # Ensure that the directory structure exists, creating it if necessary
      os.makedirs(os.path.dirname(filename), exist_ok=True)

      try:
          file.save(filename)
      except Exception as e:
          return jsonify({"error": f"Error saving file: {e}"}), 500

      # Process the image and return the prediction
      try:
          prediction = process_image(filename)
          return jsonify(prediction)
      except Exception as e:
          return jsonify({"error": f"Error processing image: {e}"}), 500
      finally:
          # Delete the image file after processing
          try:
              os.remove(filename)
          except Exception as e:
              print(f"Error deleting file: {e}")
    
  #Endpoint: predict
  @app.route('/predict/v2', methods=['POST'])
  def predict_v2():

    api_key = request.form['api_key']

    # Verify the API key
    if api_key != SECRET_API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
        
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Get the current script directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Define the subdirectory (TMP_STORAGE_PATH)
    subdirectory = "TMP_STORAGE_PATH"

    # Create the full path for the filename by joining the current directory, subdirectory, and the unique identifier with the file extension
    filename = os.path.join(current_directory, subdirectory, str(uuid.uuid4()) + '.png')

    # Ensure that the directory structure exists, creating it if necessary
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        file.save(filename)
    except Exception as e:
        return jsonify({"error": f"Error saving file: {e}"}), 500

    # Process the image and return the prediction
    try:
        cropped_row=image_processing_v2(filename)
        if cropped_row is None:

            prediction= {"prediction": None}
        else:
            prediction={"prediction": process_image_v2(cropped_row)}

        return jsonify(prediction)

    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500
    finally:
        # Delete the image file after processing
        try:
            os.remove(filename)
        except Exception as e:
            print(f"Error deleting file: {e}")

  #BeforeRequest 
  @app.before_request
  def before_request_func():
    execution_id = uuid.uuid4()
    g.start_time = time.time()
    g.execution_id = execution_id

    print(g.execution_id, "ROUTE CALLED ", request.url)

  #AfterRequest
  @app.after_request
  def after_request(response):
    if response and response.get_json():
        data = response.get_json()

        data["time_request"] = int(time.time())
        data["version"] = config.VERSION

        response.set_data(json.dumps(data))

    return response
  
  return app
  
app = create_app()

#Functions
def extract_viewed_number(text):
    # Define a regular expression pattern to match "Viewed by {number}"
    pattern = r'Viewed by (\d+)'

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the number
    if match:
        return match.group(1)
    else:
        return None

def process_image(image):
  
    # Perform OCR on the image
    result = reader.readtext(image)

    # Extract the recognized text
    text = ' '.join([entry[1] for entry in result])

    # Extract the number from the text
    viewed_number = extract_viewed_number(text)

    return {"prediction": viewed_number}

def process_image_v2(image):

    # Perform OCR on the image
    result = reader.readtext(image)

    # Extract the recognized text
    text = ' '.join([entry[1] for entry in result])

    # Extract the number from the text
    viewed_number = extract_number_after_zero_v2(text)


    if viewed_number=="" or viewed_number is None:
        return None
    else:
        return viewed_number

def image_processing_v2(filename):
    # Load the small logo image and the larger image
    logo = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)
    large_image = cv2.imread(filename)
    
    # Ensure the larger image is always in (1170, 540, 3) shape and convert it to grayscale
    large_image = cv2.resize(large_image, (540, 1170))
    gray_large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    
    # Apply template matching
    result = cv2.matchTemplate(gray_large_image, logo, cv2.TM_CCOEFF_NORMED)
    
    # Set a threshold to get the coordinates of matches
    threshold = 0.8
    loc = np.where(result >= threshold)
    
    # Check if any matches are found
    if loc[0].size > 0:
        # Extract the last set of coordinates
        pt = (loc[1][-1], loc[0][-1])
    
        # Crop the image based on the row where the logo exists
        cropped_row = large_image[pt[1]:pt[1] + logo.shape[0], loc[::-1][0][-1]:]
    
        return cropped_row
    else:
       return None 

def extract_number_after_zero_v2(input_string):
    # Define the regex pattern
    pattern = re.compile(r'0 (\d+)')
    
    # Use the pattern to find the match
    match = pattern.search(input_string)

    # Check if there's a match
    if match:
        # Access the captured number
        matched_number = match.group(1)
        return matched_number
    else:
        return None  # Return None if no match is found
        
if __name__ == "__main__":
  #    app = create_app()
  print(" Starting app...")
  app.run(host="0.0.0.0", port=5000)
