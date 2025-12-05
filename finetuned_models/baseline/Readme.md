Dataset Image Counts

1. MVTec AD
Total images: 5,354
Train: ~3,629
Test: ~1,725
Categories: 15 objects (capsule, hazelnut, screw, etc.)


2. MVTec 3D-AD
Total samples: ~4,000+ 3D scans
Categories: 10 object types
Includes RGB images, depth maps, and point clouds for each sample.


3. VisA Dataset
Total images: 10,821
Categories: 12 (various real-world food and object types)


4. BTAD (BeanTech Anomaly Detection Dataset)
Total images: 2,540
Categories: 3 industrial products
Contains both normal and defective samples.


5. MPDD (Multi-Product Defect Detection Dataset)
Total images: ~1,346
Normal (train): 888
Test (normal + anomalous): 458
Categories: Multiple product types


6. Real-IAD
Total images: ~150,000
Categories: 30 objects
Includes multi-view RGB images and anomaly annotations.


### Combined Dataset Size
Approx. Total Across All Sources:
~174,000+ images
1,00,000 are normal images across all categories and across all the sources are used for building dict model.
rest are used for testing(anomaly images)



### Model
Backbone: DinoV2
arch:Vit-Base(86 million parameters)
pretrained on LAION dataset, 1billion plus images.

Dictionary memory:
Two projection layers(for keys and values)
Two layers(keys and values)


Results:-
test_data across all sources
precision:0.3
recall:0.43


Realiad PCB test_data(including both normal and anomalous images)
threshold:0.8,
precision: 0.5
recall: 0.65

