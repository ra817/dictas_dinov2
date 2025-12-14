
Backbone:
DinoV2-base variant(86 million parameters)
Given an input image it is producing feature map of size(1369,768) where the 1369 is patch count & 768 embedding size
This feature map that we have got is  very high in semantic(as it trained upon billions of images and millions of categories)
So, the variance among the patches would be very high(the cosine similarity b/w two adjacent patches, even if they are looking same)
To reduce the variance of incoming feature map from dinoV2 backbone
We will project feature map into pcb specific domain space
This will make the key_gen and val_gen to be learned about pcb domain by removing irrelevant details.
Without this, the output of dinov2 can add noise or lot of extra details which are relevant for dictionary building
and create alot of false positive.


Dictionary layers:
firstly added two projection layers of keys and values.
MLP layers: projecting incoming 768-embedding size into 1024 embedding dimension
droput layers is added, to avoid if model is started learning or memorizing data point instead of making generalization.
then again projecting the embedding of 1024 back to 768 dimension

we have added two layers of dictionary: Global and pcb specific dictionary.
both are Intialized with dict size 0



When training starts:
First few epochs(0-3) only the projection layers will be trained with gradient updates


Then after 3rd epoch:
we start building the global as well as the pcb dict.
each patch is being matched or compared with other patch of one image with some threshold.
like for global dict insertion we have added threshold of 0.3
    if incoming patch matched with dictionary keys with more than equal to 30%, then we will merge that patch into that dictionary keys. If not, we will insert it as new keys and values in the global dict.
And same for pcb level dict, where we have taken 0.7 as threshold for merging and inserting new data points.

the global dict used to information about basic stuffs of the any pcb board.


