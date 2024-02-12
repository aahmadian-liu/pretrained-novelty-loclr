import torch
import dino.utils as utils
import dino.vision_transformer_c as vits

def load_full_checkpoint(path,backbone,norm_last_layer):

    model = utils.MultiCropWrapper(backbone, vits.DINOHead(
        in_dim=backbone.embed_dim,
        out_dim=65536,
        bottleneck_dim=256,
        use_bn=False,
        norm_last_layer=False,
    ))
    if norm_last_layer:
        model.head.last_layer = torch.nn.Linear(256, 65536, bias=False)
    else:
        model.head.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(256, 65536, bias=False))
    model.eval()
    state_dict = torch.load(path, map_location="cpu")
    state_dict = state_dict['teacher']
    msg = model.load_state_dict(state_dict, strict=False)

    return model