### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_effnetb1_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["pizza", "steak", "sushi"]

### 2. Model and transforms preparation ###

# Create EffNetB2 model
effnetb1, effnetb1_transforms = create_effnetb1_model(num_classes=len(class_names) )

# Load saved weights
effnetb1.load_state_dict(torch.load(f="pretrained_effnetb1_feature_extractor_pizza_steak_sushi_20_percent.pth",
                                    map_location=torch.device("cpu"),))
### 3. Predict function ###
# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """
    Transforms and performs a prediction on img.
    :param img: target image .
    :return: prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb1_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb1.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb1(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB1 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "I will add it soon wait.."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
