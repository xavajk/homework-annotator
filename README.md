### **Note: this was my submission for the intern competition.**

# EdLight ML Intern Take Home Task

# Understanding the Problem

This problem focuses on designing and implementing a machine learning model that takes in student’s written work as input and outputs a caption describing the students work. As an aid to educators, it is important to accurately analyze the content of students work with concise, descriptive captions.

# Understanding the Data

The domain of the data for this project is separated into two sections: the first is a CSV file containing decriptions and their corresponding images of student work; the second is a folder containing all the images of student work. With respect to the descriptions, there are 4516 examples that can be used to train, validate, and test the model. In the case of the images, there are similarly 4516 examples of student work. Each image has a corresponding description such that they are paired via the image’s filename.

The descriptions of the images vary in length but all are consistently descriptive and appropriate for the content of the image (i.e. noting table values or a shape’s side lengths). It should also be noted that some descriptions contain LaTex style equations where appropriate. The images themselves vary in content, ranging from drawings of shapes to tables to equations.

# Approach

To begin, it is important to understand the context in which the model is being designed. The two sections above aim to accomplish this. Taking a step back, the ultimate goal of this model is to receive a student’s work as input and output an appropriate description. To accomplish this, relevant research takes us in the direction of the field of Image Captioning. Image captioning is an emerging field at the intersection of computer vision and natural language that has progressed greatly in recent years. This is due to both larger CV datasets and thus larger CV models, as well as, more sophisticated NLP paradigms; i.e. the transformer.

## Architecture

There are two commonly implemented architectures for image captioning: the first utilizes a CNN, a fully-connected layer, and finally an RNN with attention; the second similarly uses a CNN followed by a transformer encoder-decoder network. Transformers have shown a lot of potential across a wide range of NLP tasks so I will be implementing the second architecture. It should be noted that RNNs (with certain modifications, e.g. [RWKV](https://huggingface.co/blog/rwkv)) can achieve better performance than traditional transformers where large context windows are necessary but for the purpose of generating short, descriptive captions for students’ work, a traditional transformer should be sufficient.

As a guide, several documentation sources were reference to guide the design of the architecture. The major changes to the architecture are the image/caption processing, head

Finally, it would be prudent to test different architectures or variations of the same architecture to establish which methods are optimal for this specific task, however given a shorter timeline, using past research to guide design will speed up this process. 

## Training

The CNN chosen for this task is EfficientNet which has benchmarked among the top of CNN models for the Imagenet dataset. It is pre-trained and will be used to present the images to the transformer numerically (the image features). The transformer encoder takes the image features and generates a new representation of the inputs. Finally the transformer decoder takes the encoder outputs along with the image captions to try and learn how to generate novel captions for that image.

This architecture will be pre-trained on the Flicker8k/30k dataset and fine-tuned on the EdLight dataset. I believe pre-training will be necessary for the model to first learn how to caption images effectively and then specialize to the students’ work. 

# Results

## Pre-Training

The first model that was trained largely used hyperparameters that were provided by [https://keras.io/examples/vision/image_captioning/](https://keras.io/examples/vision/image_captioning/), the keras documentation for implementing an image captioning model. This includes one and two multi-head attention heads for the transformer encoder and decoder respectively. Additionally, a training/validation split of 80% is used, 512 for the dimensions of the image and text embeddings, a batch size of 64, and a training cycle of 30 epochs. It was found that the model began to overfit past 10-15 epochs. (`img_cap_model_orig_12.keras`)

Thus, the next step in trying to improve the model was to one, increase the training set size, and two, to add regularization to the transformer network (Note: Dropout is already implemented in the transformers encoder and decoder). The training/validation split was increased to 90%, and then two cycles of training were performed: one with regularization and one without. (`img_cap_model_10_train_90_wreg.keras` and `img_cap_model_15_train_90.keras` respectively)

Finally in an attempt to increase the quality of the captions, the number of attention heads for the encoder/decoder were increased by one and two, resulting in two and four heads for the encoder and decoder respectively, trained for 13 epochs, using Flicker30k, the 30K examples version of Flicker8k. (`30k_img_cap_model_013_heads_2_4.keras`)

From this, it was decided that the model with the increased number of attention heads without regularization was the most performative. Of course, when any hyperparameter is changed, this leaves room for more testing and evaluation. So, with more time this should be investigated.

## Fine-Tuning

First, the ImageCaptioningModel was trained solely on the EdLight dataset to establish a baseline for performance. In theory, the fine-tuned model with pre-training on Flickr30k should be more performative.

Here are some sampled captions for the EdLight trained model which was trained for 40 epochs before EarlyStopping interrupted (`img_cap_edlight_040.keras`):

![img_cap_edlight_040_2](https://github.com/user-attachments/assets/88325105-f811-4050-8ab1-60afdf88267e)
![img_cap_edlight_040_1](https://github.com/user-attachments/assets/3a93b071-89cb-4960-8359-7e19fca3cebf)


Next, the fine_tuned model was trained (with weights initialized to that of `30k_img_cap_model_013_heads_2_4.keras`) for 34 epochs before EarlyStopping interrupted with the following results (`img_cap_edlight_034_ft.keras`):

![img_cap_edlight_ft_034_1](https://github.com/user-attachments/assets/7a645a8b-cd9b-4f85-b4a1-6a0605dda006)
![img_cap_edlight_ft_034_2](https://github.com/user-attachments/assets/9bcdc47e-1987-44cb-afa6-bffb55bb1acd)

These captions were somewhat cherry-picked to present the better generations. Both models seem to perform relatively well for their limited training however the fine-tuned model predicted captions that seem closer to the training data.

# Improvements

First and to reiterate, it would be prudent to explore non-transformer-based architecture such as the CNN + RWKV as exampled previously. Given more time, this certianly should be evaluated. Second, in order to achieve a better baseline for captions, the pre-training should be done on a much larger dataset for a longer period of time. Thirdly, there should be a more in-depth analysis of the affect the hyperparameters have on the performance, such as the batch size, number of attention heads, learning rate, etc. 

Though the pre-trained model produces coherent captions for the most part, there is still much room for improvement. Then, the performance of the fine-tuning suffers from non-optimal pre-trained weights. Ideally, the pre-trained model should be optimized ******before****** fine-tuning. In addition, using a pre-trained image captioning model (having already been trained on image-caption pairs) such as the **[AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)** with [**Microsoft GIT**](https://huggingface.co/microsoft/git-base) pre-trained weights rather than the custom model could, and most likely will, result in better performance.

Another possible avenue for increased performance lies in the quality of the images. Some of the student’s work is completely illegible so the model has a hard time learning how to interpret the data. The fine-tuned model was shown to produce `[ILLEGIBLE]` or `[NOT READABLE QUALITY]` on some occasions for which the argument can be made that the model has learned when it can’t properly decipher something which is important.

This task was really interesting for me as I got to implement a multi-modal network architecture that combines computer vision and natural language processing, something that I haven’t done before in my personal projects and studies. I see a lot of opportunity to improve this project and I look forward to spending time to improve its performance.

# Resources

[https://keras.io/examples/vision/image_captioning/](https://keras.io/examples/vision/image_captioning/)

[https://huggingface.co/docs/transformers/main/tasks/image_captioning](https://huggingface.co/docs/transformers/main/tasks/image_captioning)

[https://www.kaggle.com/datasets/adityajn105/flickr30k/code](https://www.kaggle.com/datasets/adityajn105/flickr30k/code)

[https://medium.com/mlearning-ai/understanding-efficientnet-the-most-powerful-cnn-architecture-eaeb40386fad](https://medium.com/mlearning-ai/understanding-efficientnet-the-most-powerful-cnn-architecture-eaeb40386fad)

## Model Files

`img_cap_model_orig_12.keras` : the ImageCaptioningModel trained on the Flickr8k dataset for 12 epochs

`img_cap_model_15_train_90.keras` : the ImageCaptioningModel trained on the Flickr8k dataset for 15 epochs with a 90% training split

`img_cap_model_10_train_90_wreg.keras` : the same as above but with regularization in the Dense layers trained for 10 epochs

`30k_img_cap_model_013_heads_2_4.keras` : the ImageCaptioningModel trained on the Flickr30k dataset for 13 epochs with increased `num_heads`

`img_cap_edlight_040.keras` : the ImageCaptioningModel trained on the EdLight dataset for 40 epochs

`img_cap_edlight_034_ft.keras` :  the final fine-tuned ImageCaptioningModel trained on the EdLight dataset for 34 epochs using `30k_img_cap_model_013_heads_2_4.keras`’s weights upon initialization

## Data Files

`descriptions.csv` : the provided caption-image pair CSV file

`edlight_images.zip` : the zipped folder of EdLight supplied images

`flick30k.zip` : the Flickr30k dataset zipped folder
