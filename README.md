# **Multimodal Learning by Combining FMS Image Representations With Wireline Logs Using Custom Feed-Forward Neural Network Model for Final Classification**
### This repository contributes to Chapter 6 of Saira Baharuddin's PhD thesis, entitled "Multimodal Learning for Carbonate Rock Classification Using Microresistivity Images and Wireline Logs".

## Brief Overview
This is my third approach as part of the multimodal learning for carbonate rock classification where I incorporate both FMS images and  wireline log data. A custom feed-forward neural network model consisting of densely connected layers was appended to the concatenated layer. The first layer is a dense layer with 256 units and a ReLU activation function, acting as the input layer with the shape determined by the number of features in the combined dataset. Following this, a dropout layer is included with a 
 dropout rate of 0.5 to prevent overfitting by randomly deactivating 50% of the neurons during training. Subsequently, another dense layer with 128 units and a ReLU
activation function is added, followed by another dropout layer. Lastly, a dense layer with 64 units and a ReLU activation function is appended before the final output layer, which consists of 5 units representing the number of Dunham classes used in my study. The output layer uses the SoftMax activation function to output probabilities for each class.The model is compiled using the Adam optimizer with a learning rate of 1E-4, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.
Furthermore, various callbacks, including early stopping and model checkpointing, are employed to monitor the training process and prevent overfitting. The model is trained on the combined features dataset over 200 epochs with diVerent batch sizes. Overall, this approach was negatively affected by the features extracted from VGG19 CNN. The 512 FMS features may not be as relevant as the wireline log data for classification. They could potentially introduce noise to the models and make it harder to discern meaningful patterns for classification. Therefore, the outcome suggests that while combining modalities and using custom models can deliver advantages, factors such as model complexity, small imbalanced datasets, and heterogeneous features likely contributed to moderate performances in our case. In conclusion, although deep learning is a widely recognized approach for classification and prediction, it may not be the best choice for this small dataset. Additional analysis, fine-tuning and alternative meta-learners could improve performance.

<p align="center">
  <img width="977" alt="customNN" src="https://github.com/ssaira267/vgg19_customNN_rockclassification/assets/57672761/86fb8f79-061f-48a1-ae7e-0148c8e626bc">
Network architecture of the multimodal learning using the VGG19 model with feed-forward neural networks with fully connected layers for rock classification.
</p>

## Repository Contents
- Data: Sample datasets used for training and testing the models are publicly available on the following website: https://mlp.ldeo.columbia.edu/logdb/scientific_ocean_drilling. The dataset used are from Leg 194, holes 1194B, 1196A and 1199A. The .csv files inside the data directory serve as examples demonstrating how the data is arranged and used for this study. These files contain structured data in tabular format, where each row represents a sample or observation based on depth, and each column represents a feature or attribute of that sample. Due to the large file size of the FMS images, I am unable to upload the FMS images folder. However, the images are available on the website mentioned earlier. It's important to note that the FMS images have been cropped based on the depth intervals specified for the study.
- Notebooks: Python codes developed include data preprocessing, model training, model evaluation, and visualization of the results.
- Documentation: Additional resources and documentation that outline the methods used in this study.

## How to Use
To run the scripts in this repository:
1. Ensure that Python 3.8 (and above) and the necessary libraries (listed in requirements.txt) are installed.
2. Clone the repository to your local machine.
3. Navigate to the notebooks directory and run the desired notebook as follows:
  
```bash
  python notebook_name.py
```

## Dependencies 
- Dlisio
- Imbalanced-learn
- Livelossplot
- Matplotlib
- NumPy
- OpenCV-python
- pandas
- Scikit-learn
- Seaborn
- Tensorflow
- Tensorflow-Keras
- XGBoost

Please refer to 'requirements.txt' for a complete list of dependencies.

## Contact
For any queries related to this repository or the research, please contact:
- Saira Baharuddin                  - Email: s.baharuddin19@imperial.ac.uk
- Prof Cédric John (PhD Supervisor) - Email: cedric.john@qmul.ac.uk
