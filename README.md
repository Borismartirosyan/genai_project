# Consistent Character Generation with Stable Diffusion XL + GPT-4o Image Generation

The project is created to understand, which method, GPT-4o, InstandID (SDXL Realistic Vision 5 checkpoint) and Dreambooth (SDXL Realistic Vision 5 checkpoint) better is in consistend character generation task. For this, we gathered images of several celebrities (see the train_dataset_dreambooth for full training datapoints) and used GPT-4o UI, InstandID and Dreambooth notebook to generate images. Then, we compared their CLIP embeddings, of these images, with Cosine Similarity and Wasserstain Distance.

To configure project clone the repo and follow steps
``` 
python3 -m venv venv
```
Then
```
source venv/bin/activate
```
After that
```
pip3 install -r requirements.txt
```

To train the Dreambooth and see your results, please see train/dreambooth folder
To use InstandID inference, please use train/instant_id folder
To use GPT-4o, you know where to go.
