import os
import torch
import torchvision.utils as vutils

def my_validation(networks, content_loader, style_loader, args):
    my_output = args.my_output
    if not os.path.exists(my_output):
        os.makedirs(my_output)

    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module

    C_EMA.eval()
    G_EMA.eval()

    # ones = torch.ones(1, content_sample.size(1), content_sample.size(2), content_sample.size(3)).cuda(args.gpu, non_blocking=True)
    ones = None
    # x_res_ema = torch.cat((ones, content_sample.cuda(args.gpu, non_blocking=True)), 0)
    x_res_ema = None
    content_list = []
    for _, imgs in enumerate(content_loader):
        content_list.append(imgs)
        if ones is None:
            ones = torch.ones(1, imgs.size(1), imgs.size(2), imgs.size(3))
        if x_res_ema is None:
            x_res_ema = torch.cat((ones, imgs), 0)
        else:
            x_res_ema = torch.cat((x_res_ema, imgs), 0)
    
    nrows = x_res_ema.size(0)
    # improtant flag when validation or testing
    with torch.no_grad():
        for _, style_imgs in enumerate(style_loader):
            x_ref = style_imgs
            batch_style = x_ref.size(0)
            # nrows += batch_style
            for i in range(batch_style):
                x_res_ema = torch.cat((x_res_ema, x_ref[i:i+1]), 0)
                for content_imgs in content_list:
                    x_src = content_imgs.cuda(args.gpu, non_blocking=True)
                    x_ref_tmp = x_ref[i:i+1].repeat((content_imgs.size(0), 1, 1, 1)).cuda(args.gpu, non_blocking=True)

                    c_src, skip1, skip2 = G_EMA.cnt_encoder(x_src)
                    s_ref = C_EMA(x_ref_tmp, sty=True)
                    x_res_ema_tmp, _ = G_EMA.decode(c_src, s_ref, skip1, skip2)

                    x_res_ema_tmp = x_res_ema_tmp.cpu()
                    x_res_ema = torch.cat((x_res_ema, x_res_ema_tmp), 0)

        model_name = os.path.basename(args.res_dir)
        vutils.save_image(x_res_ema, os.path.join(my_output, '{}_EMA_custom.jpg'.format(model_name)), normalize=True, nrow=nrows)
    