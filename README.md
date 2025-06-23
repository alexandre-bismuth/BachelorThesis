# Optimising Gunshot Detection in Tropical Forests for Wildlife Protection with TinyML
Alexandre Bismuth, BSc in Mathematics and Computer Science of Ecole Polytechnique, France

Bachelor Thesis supervised by Professor Alex Rogers, St Anne's College, University of Oxford, United Kingdom. The full report as well as the defense slides can be found in this repository in `Bachleor_Thesis_Gunshot_Detection_TinyML.pdf` and `Bachelor_Thesis_Defense.pdf` respectively.

*All of the present code is original unless explicitly stated otherwise*

---
## Setup 

### Project Environment

This project was developed in an AWS environment through SageMaker Studio. More specifically, it ran on a JupyterLab instance on SageMaker Distribution 2.4.0 with 16GB of storage. We used an `ml.t3.medium` instance for development and an `ml.g5.2xlarge` for training. 

The specifications of these instances can be summarized in the following table: 

| Instance Name  | vCPUs | Instance Memory (GiB) | GPU Model    | Memory per GPU (GB) |
|----------------|-------|-----------------------|--------------|---------------------|
| ml.t3.medium   | 2     | 4                     | N/A (No GPU) | N/A                 |
| ml.g5.2xlarge  | 8     | 32                    | NVIDIA A10G  | 24                  |

### Data

The tropical gunshot data can be found [here](https://data.mendeley.com/datasets/x48cwz364j/3). To make it compatible with our notebooks without any path editing, you should unzip the file, rename the directory as "Data", and place it within the root of the directry.

### Dependencies

This project relies on various packages pinned to specific versions (almost always the latest) to avoid the versioning issues encountered in the previous pipeline while optimizing performance and ensuring stability. Additionally, some imports require extra shell commands to run without errors. 

To address both issues at once, we provide a lifecycle shell script that should be executed every time an instance is initialized. To run it, simply execute `./lifecycle` directly from your terminal.
