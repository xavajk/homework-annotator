# homework-annotator
A deep learning pipeline to annotate and caption written work.

## Technical Details
### Image Model
The image model is the pretrained EfficientNet B0 model. The CNN was used to get a vector representation of the image to pass to the transformer.

### Text Model
Uses the traditional encoder-decoder transformer network outline in the Attention Is All You Need paper (https://arxiv.org/abs/1706.03762). Takes a representational vector embedding of an image as input and outputs a variable length caption relating to the image. Complete implementation specs are in the notebook.

### Bringing it all Together
The input is a regular image of pre-defined resolution. This image is passed to the CNN to get a vector embedding. This embedding is used as the input to the transformer. The transformer then outputs an arbitrary-length annotation for the image. This network was pre-trained on a larger, generic image-caption pair dataset, and fine-tuned on the handwritten work to boost performance.
