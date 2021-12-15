from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss


def trainGAN(data_loader, networks, opts, epoch, args, additional):
    # avg meter
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    #########
    d_daku_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    #########
    g_daku_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_rec = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']
    #########
    D_DAKU = networks['D_jp_dakuten']
    G = networks['G'] if not args.distributed else networks['G'].module
    C = networks['C'] if not args.distributed else networks['C'].module
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module
    # set opts
    d_opt = opts['D']
    #########
    d_daku_opt = opts['D_jp_dakuten']
    g_opt = opts['G']
    c_opt = opts['C']
    # switch to train mode
    D.train()
    D_DAKU.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    logger = additional['logger']

    # summary writer
    train_it = iter(data_loader)

    t_train = trange(0, args.iters, initial=0, total=args.iters)

    for i in t_train:
        #########
        try:
            imgs, style_imgs, y_org, jp_daku_class = next(train_it)
        except:
            train_it = iter(data_loader)
            imgs, style_imgs, y_org, jp_daku_class = next(train_it)
        
        x_org = imgs
        x_org = x_org.cuda(args.gpu)
        y_org = y_org.cuda(args.gpu)
        jp_daku_class = jp_daku_class.cuda(args.gpu)

        x_ref_idx = torch.randperm(x_org.size(0))
        x_ref_idx = x_ref_idx.cuda(args.gpu)

        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]

        if args.use_stn:
            x_style_org = style_imgs
            x_style_org = x_style_org.cuda(args.gpu)
            x_style_ref = x_style_org.clone()
            x_style_ref = x_style_ref[x_ref_idx]

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################
        with torch.no_grad():
            y_ref = y_org.clone()
            y_ref = y_ref[x_ref_idx]
            #########
            y_daku_ref = jp_daku_class.clone()
            y_daku_ref = y_daku_ref[x_ref_idx]
            if args.use_stn:
                rectify_x_style_ref = G.cnt_encoder.rectify(x_style_ref)
                s_ref = C.moco(rectify_x_style_ref)
                c_src, skip1, skip2 = G.cnt_encoder(x_style_org)
                x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)
            else:
                s_ref = C.moco(x_ref)
                c_src, skip1, skip2 = G.cnt_encoder(x_org)
                # x_fake = G.decode(c_src, s_ref, skip1, skip2)
                x_fake, _ = G.decode(c_src, s_ref, skip1, skip2)

        if args.use_stn:
            # x_style_ref.requires_grad_()
            # rectify_x_style_ref.requires_grad_()
            # Discriminator
            # d_real_logit, _ = D(rectify_x_style_ref, y_ref)
            d_real_logit, _ = D(x_ref, y_ref)
            # d_real_style_logit, _ = D(x_style_ref, y_ref)
            d_fake_logit, _ = D(x_fake.detach(), y_ref)
            # Discriminator loss
            d_adv_real = calc_adv_loss(d_real_logit, 'd_real') # + calc_adv_loss(d_real_style_logit, 'd_real')
            d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')
            ######### Dakuten Discriminator
            d_daku_real_logit, _ = D_DAKU(x_ref, y_daku_ref)
            # d_daku_real_style_logit, _ = D_DAKU(x_style_ref, y_daku_ref)
            d_daku_fake_logit, _ = D_DAKU(x_fake.detach(), jp_daku_class)
            ######### Dakuten Discriminator loss
            d_daku_adv_real = calc_adv_loss(d_daku_real_logit, 'd_real') # + calc_adv_loss(d_daku_real_style_logit, 'd_real')
            d_daku_adv_fake = calc_adv_loss(d_daku_fake_logit, 'd_fake')

            #########
            d_adv = d_adv_real + d_adv_fake
            d_daku_adv = d_daku_adv_real + d_daku_adv_fake

            # d_gp = args.w_gp * compute_grad_gp(d_real_logit, rectify_x_style_ref, is_patch=False)
            #########
            # d_daku_gp = args.w_gp * compute_grad_gp(d_daku_real_style_logit, x_style_ref, is_patch=False)
            d_loss = d_adv + d_daku_adv

            d_opt.zero_grad()
            d_adv_real.backward(retain_graph=True)
            # d_gp.backward()
            d_adv_fake.backward()
            if args.distributed:
                average_gradients(D)
            d_opt.step()

            #########
            d_daku_opt.zero_grad()
            d_daku_adv_real.backward(retain_graph=True)
            # d_daku_gp.backward(retain_graph=True)
            # d_daku_gp.backward()
            d_daku_adv_fake.backward()
            d_daku_opt.step()
        else:
            x_ref.requires_grad_()
            d_real_logit, _ = D(x_ref, y_ref)
            d_fake_logit, _ = D(x_fake.detach(), y_ref)

            d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
            d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')
            #########
            d_daku_real_logit, _ = D_DAKU(x_ref, y_daku_ref)
            d_daku_fake_logit, _ = D_DAKU(x_fake.detach(), jp_daku_class)
            #########
            d_daku_adv_real = calc_adv_loss(d_daku_real_logit, 'd_real')
            d_daku_adv_fake = calc_adv_loss(d_daku_fake_logit, 'd_fake')
            #########
            d_adv = d_adv_real + d_adv_fake
            d_daku_adv = d_daku_adv_real + d_daku_adv_fake

            d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False)
            #########
            d_daku_gp = args.w_gp * compute_grad_gp(d_daku_real_logit, x_ref, is_patch=False)

            d_loss = d_adv + d_gp + d_daku_adv + d_daku_gp

            d_opt.zero_grad()
            d_adv_real.backward(retain_graph=True)
            d_gp.backward()
            d_adv_fake.backward()
            if args.distributed:
                average_gradients(D)
            d_opt.step()

            #########
            d_daku_opt.zero_grad()
            d_daku_adv_real.backward(retain_graph=True)
            # d_daku_gp.backward(retain_graph=True)
            d_daku_gp.backward()
            d_daku_adv_fake.backward()
            d_daku_opt.step()

        # Train G
        if args.use_stn:
            with torch.no_grad():
                rectify_x_style_org = G.cnt_encoder.rectify(x_style_org)
                rectify_x_style_ref = G.cnt_encoder.rectify(x_style_ref)
            s_src = C.moco(rectify_x_style_org)
            s_ref = C.moco(rectify_x_style_ref)
            c_src, skip1, skip2 = G.cnt_encoder(x_style_org)
        else:
            s_src = C.moco(x_org)
            s_ref = C.moco(x_ref)
            c_src, skip1, skip2 = G.cnt_encoder(x_org)
        # x_fake = G.decode(c_src, s_ref, skip1, skip2)
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        # x_rec = G.decode(c_src, s_src, skip1, skip2)
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        g_fake_logit, _ = D(x_fake, y_ref)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        # g_daku_fake_logit, _ = D_DAKU(x_fake, y_daku_ref)
        g_daku_fake_logit, _ = D_DAKU(x_fake, jp_daku_class)
        g_daku_rec_logit, _ = D_DAKU(x_rec, jp_daku_class)
        
        g_daku_adv_fake = calc_adv_loss(g_daku_fake_logit, 'g')
        g_daku_adv_rec = calc_adv_loss(g_daku_rec_logit, 'g')

        g_adv = g_adv_fake + g_adv_rec
        g_daku_adv = g_daku_adv_fake + g_daku_adv_rec

        g_imgrec = calc_recon_loss(x_rec, x_org)

        c_x_fake, _, _ = G.cnt_encoder(x_fake)
        g_conrec = calc_recon_loss(c_x_fake, c_src)
        # Style image and original image latent code should be the same.
        # if args.use_stn:
        #     c_orig_src, _, _ = G.cnt_encoder(x_org)
        #     g_conrec += calc_recon_loss(c_orig_src, c_src)
        #     # g_conrec += calc_recon_loss(c_orig_src, c_x_fake)
        #     g_conrec *= (1/2)

        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec +args.w_rec * g_conrec + args.w_off * offset_loss + args.w_daku_adv * g_daku_adv

        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()
        if args.distributed:
            average_gradients(G)
            average_gradients(C)
        c_opt.step()
        g_opt.step()

        ##################
        # END Train GANs #
        ##################


        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_daku_advs.update(d_daku_adv.item(), x_org.size(0))
                if not args.use_stn:
                    d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_daku_advs.update(g_daku_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

                moco_losses.update(offset_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/ADV_DAKU', d_daku_advs.avg, summary_step)
                if not args.use_stn:
                    add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/ADV_DAKU', g_daku_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                add_logs(args, logger, 'C/OFFSET', moco_losses.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(epoch + 1, args.epochs, i+1, args.iters,
                                                        training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)

