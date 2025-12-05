### Fine tuning on top of baseline using the PCB normal iamges only

Dataset Image Counts
1. Real-IAD
Total images: 14,000(7000 normal images)
Categories: 1 objects(pcb)
Includes multi-view RGB images and anomaly annotations.


### Model
Backbone: DinoV2
arch:Vit-Base(86 million parameters)
pretrained on LAION dataset, 1billion plus images.

Dictionary memory:
Two projection layers(for keys and values)
Two layers(keys and values)



Realiad PCB test_data(including both normal and anomalous images)
threshold:0.8,
precision: 0.6
recall: 0.75

