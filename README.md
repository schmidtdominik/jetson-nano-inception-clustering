# Image clustering based on InceptionV3 activations on an Nvidia Jetson Nano

The goal of this project is to finely cluster images of various martian surface features (like rock formations, sand dunes, ravines, etc.). This may allow Tumbleweeds operations team to more selectively decide which images to transmit back to Earth from Mars to better utilize the available bandwidth.

The underlying clustering technique is the basic (but fast) k-means algorithm. K-means clustering works by starting with some initial (random or heuristically selected) cluster centers and then adjusting them iteratively.

Unfortunately k-means performs badly in high-dimensional spaces (such as most real-world images) where datapoints are located on complex submanifolds. Thus applying k-means directly to images usually doesn't work very well. Especially in cases where different clusters can't easily be distinguished through colors or image composition.

To remedy this we first pass the images through all but the last layer of a pretrained InceptionV3 neural network.
This acts as a dimensionality reduction technique which greatly simplifies the clustering task itself. The network produces semantically meaningful outputs without needing to be trained on our dataset specifically.
Finally we cluster based on the neural activations at that layer.

Sidenotes:
The InceptionV3 network was pretrained on the imagenet dataset. It can be replaced by any larger (and possibly smaller) architectures such as Resnets while maintaining a similar performance.
K-means could be replaced by other fast clustering algorithms. For the initial training step (where we determined the cluster centers) we clustered about 30000 images, each with 2048 activation dimensions. Any number of clusters between 32 and 512 yielded useful results.

Example clustering with 128 clusters and using [this dataset](https://dominikschmidt.xyz/mars32k/):

**Cluster 0**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c1.png)

**Cluster 1**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c2.png)

**Cluster 2**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c3.png)

**Cluster 3**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c4.png)

**Cluster 4**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c5.png)

**Cluster 5**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c6.png)

**Cluster 6**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c7.png)

**Cluster 7**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c8.png)

**Cluster 8**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c9.png)

**Cluster 9**

![Image](https://raw.githubusercontent.com/schmidtdominik/jetson-nano-inception-clustering/master/example%20images/c10.png)
