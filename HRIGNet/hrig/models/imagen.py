import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from einops import rearrange, repeat

from ldm.util import default,instantiate_from_config

from imagen_pytorch import Unet, Imagen
import ipdb



class ImagenHRIG_Uncond(Imagen):
    def __init__(self,
                 unet_configs,
                 *args, **kwargs):
        unets = []
        for unet_config in unet_configs:
            unet = instantiate_from_config(unet_config)
            unets.append(unet)
        super().__init__(unets=unets,*args, **kwargs)




class ImagenHRIG_PL(pl.LightningModule):
    def __init__(self,
                 imagen_config,
                 image_key="image_gt",
                 training_unet_number=1,
                 *args, **kwargs):
        super().__init__()
        self.imagen = instantiate_from_config(imagen_config)
        self.image_key = image_key
        self.training_unet_number = training_unet_number


        ckpt_path = kwargs.pop("ckpt_path", None)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)


    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        

    def configure_optimizers(self):
        lr = self.learning_rate
        unet = self.imagen.unets[self.training_unet_number-1]
        params = list(unet.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        
        return opt
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def shared_step(self, batch):
        x = self.get_input(batch,self.image_key)
        loss = self.imagen(x, unet_number = self.training_unet_number)

        prefix = 'train' if self.training else 'val'
        loss_dict = dict()
        loss_dict.update({f'{prefix}/loss_imagen': loss})
        return loss,loss_dict
    
    def training_step(self, batch, batch_idx):
        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
            logger=True, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss,loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=False, 
            logger=True, on_step=False, on_epoch=True)

        return loss
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch_size = batch[self.image_key].shape[0]
        outputs = self.imagen.sample(batch_size = batch_size,return_all_unet_outputs=True) # (B, C, H, W)

        for index in range(len(self.imagen.image_sizes)):
            log["samples_unet"+str(index+1)] = outputs[index]
        return log