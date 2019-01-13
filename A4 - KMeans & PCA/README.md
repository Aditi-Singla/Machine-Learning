## Assignment 3 - KMeans and Principal Component Analysis

This assignment involves implementation of KMeans Clustering algorithm for recognizing human activities and postural transitions
(based on attributes collected through a smartphone) and Principal Component Analysis algorithm for face recognition. Statement for the same is linked [here](./Statement.pdf).

A detailed report with experimentation and results can be found [here](./Report.pdf).

#### Running the code

To run the algorithms -

- For q1 on KMeans,

  ```bash
  python q1.py <attributes> <labels>
  ```
  The dataset that has been used here uses the command:

  ```bash
  python q1.py data/q1_kmeans/attr.txt data/q1_kmeans/label.txt
  ```

- For q2 on PCA, 

  ```bash
  python q2.py
  ```
  This runs on datasets for LFW and ORL faces ([link](./data/q2_pca/)), and generates the eigen faces for each of the two datasets, along with several temp files that are used for image reconstruction. Now to run image reconstruction on any of the images in the given dataset,

  ```bash
  python q2_reconstruct.py <image-address>
  ```
  Note: The image address should be of the following format: './data/q2_pca/orl_faces/s1/1.pgm' (for example).
