# Explainable approach for defect detection using computer vision

This project presents a system to classify defective or non-defective products based on computer vision techniques with an explainable approach. The system also returns co-ordinates of defective region within the given image. 
The dataset provided for training includes two sets of images: good and anomaly. For defective regions in anomaly images, no ground truths or labels for regions are provided. System is able to identify the defective part in anomaly image without any specific information provided. This is achieved with the help of feature maps of the convolutional layers and a particular neural network architecture described below. 


## Architecture:

1. First, take the VGG19 ImageNet model without its top layers, i.e., without the 3 FC layers at the top.
2. Then add the Global Average pooling 2d layer on top.
3. Next, add a dense layer with only one unit followed by the output layer with two units for two outcomes: good and anomaly.
4.  The architecture looks like follows : 

<picture>
 <img alt="Architecture summary" src="/assets/images/summary.png">
</picture>

			
### Why this specific architecture?

This architecture makes it easy to add explainability to our model. The VGG block outputs feature maps, each representing particular information. The global average pooling layer present after the VGG block takes the average of each feature map and produces a single scalar value, which is then multiplied with the corresponding weight assigned to it by the dense layer. Thus, feature maps associated with the higher weights indicate the part as important and vice versa. Accordingly, the output is predicted as good or anomaly. 


## Implementation:  


•	The product used for this project is – Bangle. 


### Getting the defected region for images labelled as anomaly – 

•	First, retrieve all the feature maps in the last conv layer of VGG19 model, i.e., block5_conv3 layer. 

•	Then get the weights assigned by the next dense layer to each of feature map.

•	Multiply each feature map with the corresponding weight( feature maps are just arrays; perform scalar multiplication).

•	The resultant array is a heatmap of the original input image which shows the significance of different parts of the image towards the prediction.

•	Upsample the image with any technique.

•	Then retrieve the bounding box (The significant parts of the heatmap have higher values; hence apply a threshold to the heatmap array  and then get the upper left and lower right corners of the region). 



## Example :


### Input image :

<picture>
 <img alt="input image" src="/assets/images/input.png">
</picture>

 

### Resultant image :

<picture>
 <img alt="resultant image" src="/assets/images/resultant.png">
</picture>

 

### Upsampled image :

<picture>
 <img alt="upsampled image" src="/assets/images/upsampled.png">
</picture>

 

### Defective part:

<picture>
 <img alt="defective part" src="/assets/images/defective.png">
</picture>

 

### Bounding box : 

<picture>
 <img alt="final image" src="/assets/images/final.png">
</picture>

 



### References 

This project is inspired by this awesome article from Olga Chernytska : 

Explainable Defect Detection Using Convolutional Neural Networks - Case Study : https://towardsdatascience.com/explainable-defect-detection-using-convolutional-neural-networks-case-study-284e57337b59


<hr>

Note - : 
This project was made to particularly demonstrate the particular approach mentioned and not for production line. Hence, tools like machine learning pipelines, data augmentation and transformation approaches, deployment, etc. have not been used here. The production ready complete system can be developed using the tools mentioned and can then be deployed to production in industries.

