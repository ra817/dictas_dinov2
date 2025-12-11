MAE: 


feats_all[3] → output of layer 11 → shape [B, 1370, 1024]
feats_all[11] → output of layer 17 → shape [B, 1370, 1024]
feats_all[17] → output of layer 23 → shape [B, 1370, 1024]
feats_all[23] → output of layer 31 → shape [B, 1370, 1024]


The keys and the values of the dictionary are getting gradient update during model building 
we have two projection layer 





# def forward(self, imgs):
    #     #Extract patch tokens
    #     with torch.no_grad():
    #         feats_all = self.dinov2.get_intermediate_layers(
    #                 imgs, n=self.layer_indices, reshape=False, return_class_token=False)
    #         feats_proc = [F.normalize(f, dim=-1) for f in feats_all]
    #         img_feats = torch.stack(feats_proc, dim=0).mean(0)           #average across 4 layers


    #     B, N, D = img_feats.shape
    #     img_feats_flat = img_feats.reshape(B * N, D)


    #     #Dictionary reconstruction
    #     retrieved, _ = self.dictionary.lookup(img_feats_flat, self.top_k, self.lookup)
    #     retrieved = retrieved.view(B, N, -1)


    #     #Patch reconstruction loss
    #     diff = (img_feats - retrieved).pow(2).sum(-1)
    
    #     L_recon = diff.mean()


    #     #Smoothness regularization
    #     h = int(N ** 0.5)
    #     diff_map = diff.view(B, h, h)
    #     L_smooth = (
    #         F.l1_loss(diff_map[:, :, 1:], diff_map[:, :, :-1]) + 
    #         F.l1_loss(diff_map[:, 1:, :], diff_map[:, :-1, :])
    #     )


    #     #Feature alignment (global mean)
    #     img_global = F.normalize(img_feats.mean(1), dim=-1)
    #     dict_mean = F.normalize(self.dictionary.keys.mean(0, keepdim=True), dim=-1)
    #     L_align = 1 - (img_global * dict_mean).sum(-1).mean()

    #     total_loss = L_recon + self.lambda_align * L_align + self.lambda_smooth * L_smooth
    #     return total_loss, L_recon, L_align, L_smooth




    git remote add origin https://github.com/<username>/<repo>.git
    git branch -M main


====== EPOCH SUMMARY ======
Epoch 5/5
Total EMA updates this epoch: 1471675
Total NO-UPDATES this epoch: 0
Unique dictionary keys updated this epoch: 6
Keys: [8, 50, 69, 91, 142, 177]
============================

Unique dictionary keys updated this epoch: 9
Keys: [187, 129, 89, 155, 58, 18, 17, 190, 168]

Key  129 updated 7938 times
Key  190 updated 6851 times
Key   89 updated 2839 times
Key   18 updated 2602 times
Key  187 updated 2112 times
Key  155 updated 1733 times
Key   58 updated 1509 times
Key   17 updated 923 times
Key  168 updated 873 times



the Dinov2:-
when we provide a single image to this encoder we will get (1369,768).
So, we will get 768 embedding for each of the patch of 1369.
Like the dinoV2 is trained on billions of iamges.
so each patch used to contain embedding of  high variance(cosine similarity in between two patch is very high)


Dict_key_gen:- here like we need to have the key_gen because it used to move the q_featues into dictionary space


Unique dictionary keys updated this epoch: 11
Keys: [155, 168, 190, 187, 89, 17, 18, 58, 129, 110, 162]

problem:
Inference results was fluctuating for same input image.
Issue:
during inference we where loading the checkpoint using strict=False, this made some of the layers to be randomly intialized and doesnot overwrite with 
learned weights 

Solution:
1: start with dict_size:0