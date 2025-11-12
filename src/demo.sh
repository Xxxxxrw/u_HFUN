# EDSR baseline model (x2) + JPEG augmentation
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2 --reset --data_train DIV2K+DIV2K-Q75 --data_test DIV2K+DIV2K-Q75
#python main.py --model MYEDSR --scale 2 --patch_size 96 --n_resblocks 16 --res_scale 1 --n_feats 64 --lr 1e-4 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --save myedsr_x2 --epochs 300

# Test your model
#python main.py --model MYEDSR --scale 2 --test_only --data_test Set5+DIV2K --data_range 801-900 --pre_train ../experiment/myedsr_x2/model/model_latest.pt

# fmen_onlytest(train)
#python main.py --model TRAIN_FMEN --scale 2 --test_only --data_test Set5+Set14 --patch_size 128 --pre_train ../experiment/fmen_x2/model/model_best.pt
# fmen_onlytest(test重参数化)
#python main.py --model TEST_FMEN --scale 2 --test_only --data_test Set5+Set14 --patch_size 128 --pre_train ../experiment/test.pt
#fmen_train_model
#python main.py --model TRAIN_FMEN --scale 2 --batch_size 64 --patch_size 128 --data_test Set5+Set14 --lr 5e-4 --epoch=300 --save fmen_x2


#####MYMODEL train
#python main.py --model MYMODEL --scale 2 --lr 1e-4 --batch_size 32 --patch_size 96 --n_resgroups 2 --decay 200-400-600-800-900 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_2_x2
#####MYMODELTWO train
#python main.py --model MYMODELTWO --scale 2 --lr 1e-4 --batch_size 16 --patch_size 192 --decay 200-500-800-900 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --save mymodeltwo_12x2 --epochs 1000
#####MYMODELTHREE train
#python main.py --model MYMODELTHREE --scale 2 --lr 1e-4 --batch_size 32 --patch_size 96 --n_feats 48 --decay 200-400-600-800-900 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_simple2_x2
#####MYMODELv1_x2 train
#python main.py --model MYMODELV1 --scale 2 --lr 1e-4 --batch_size 32 --patch_size 128 --n_feats 64 --decay 200-400-600-800-900 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1_x2
#####MYMODELv1r_x2 train
#python main.py --model MYMODELV1R --scale 4 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1r_x4_conv_64C
#python main.py --model MYMODELV1R --scale 2 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1r_x2_conv_64C
#####MYMODELv1r_x4 train
#python main.py --model MYMODELV1R --scale 4 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1r_x4_avg_64C

#####MYMODELv1r_repeat_x2 train
#python main.py --model MYMODELV1R_REPEAT --scale 2 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1rep_skip_x2_avg_64C
#####MYMODELv1r_repeat_x4 train
#python main.py --model MYMODELV1R_REPEAT --scale 4 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1repeatnoskip_x4_avg_64C

#####MYMODELv1_x4 train
#python main.py --model MYMODELV1 --scale 4 --lr 7e-4 --batch_size 32 --patch_size 192 --n_feats 64 --decay 200-400-600-800-900 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1_x4
#####MYMODELv1_rearrange_x2 train
#python main.py --model MYMODELV1 --scale 2 --lr 7e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800-900 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v1_2222_x2
#####MYMODELv2_stu_x2 train
#python main.py --model MYMODELV2 --scale 2 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v2_stu_x2
#####MYMODELv2_stu_x4 train
#python main.py --model MYMODELV2 --scale 4 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v2_stu_x4
#####MYMODELv2_tea_x2 train
#python main.py --model MYMODELV2 --scale 2 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v2_tea_x2_48c
#####MYMODELv2_tea_x4 train
#python main.py --model MYMODELV2 --scale 4 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v2_tea_x4
#####MYMODELv2_stu_x2 train
#python main.py --model MYMODELV2 --scale 2 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v2_stu_x2_64C
#####MYMODELv3__x2 train
#python main.py --model MYMODELV3 --scale 2 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v3_x2_64C
#####MYMODELv3__x4 train
#python main.py --model MYMODELV3 --scale 4 --lr 1e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v3_x4_64C
#####MYMODELv4_x2 train
#python main.py --model MYMODELV4 --scale 2 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v4_x2_64C_conv
#####MYMODELv4_x4 train
#python main.py --model MYMODELV4 --scale 4 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v4_x4_64C_avg
#####MYMODELv5_x2 train
#python main.py --model MYMODELV5 --save mymodelv5_80.7G --scale 2 --lr 6e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv5_x4 train
#python main.py --model MYMODELV5 --scale 4 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v56block_x4_64C_avg
#####MYMODELv5_test_x2 train
#python main.py --model MYMODELV5_TEST --scale 2 --lr 3e-4 --batch_size 16 --patch_size 128 --n_feats 64 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000 --save mymodel_v5_x2_64C_dropsigaddbn
#####MYMODELv6_noshare_x2 train
#python main.py --model MYMODELV5_TESTNOSHARE --save mymodel_v6noshare_x2_48C --scale 2 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv6_share_x2 train
#python main.py --model MYMODELV6_SHARE --save mymodel_v6share_x2_48C_141G --scale 2 --lr 6e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv6_share_x3 train
#python main.py --model MYMODELV6_SHARE --save mymodel_v6share_x3_48C_115G --scale 3 --lr 6e-4 --batch_size 16 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv6_share_x4 train
#python main.py --model MYMODELV6_SHARE --save mymodel_v6share_x4_48C_115G --scale 4 --lr 6e-4 --batch_size 16 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000

#####MYMODELv6_share_x3 train
#python main.py --model MYMODELV6_SHARE --save mymodel_v6share_x3_48C_6e --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv6_share_x4 train
#python main.py --model MYMODELV6_SHARE --save mymodel_v6share_x4_48C_6e --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv7_share_x2 train
#python main.py --model MYMODELV7_SHARE --save mymodel_v7share_x2_48C_6e --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000


#####mymodel_v8_noshare_x2 train
#python main.py --model MYMODELV1_DW_6BLOCK --save MYMODELV1_DW_6BLOCK_x3_48C_DF2K_continue --scale 3 --lr 3.75e-5 --batch_size 32 --patch_size 192 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV1_DW_6BLOCK_x3_48C_DF2K/model/model_best.pt --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=32
#python main.py --model MYMODELV1_DW_6BLOCK --save MYMODELV1_DW_6BLOCK_x4_48C_DF2K_continue --scale 4 --lr 7.5e-5 --batch_size 32 --patch_size 256 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV1_DW_6BLOCK_x4_48C_DF2K/model/model_best.pt --decay 27-227 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=227
#python main.py --model MYMODELV1_DW_6BLOCK --save MYMODELV1_DW_6BLOCK_pre_x2_test --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/mymodel_v1_dwUPA_6b_32bsize_x2/model/model_best.pt --scale 2 --lr 1e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save MYMODELV8_pre_x2_test --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_x2_skip_16bszie/model/model_best.pt --scale 2 --lr 1e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save MYMODELV8_beforeatten_dropcopy0.05_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save MYMODELV8_beforeatten_dropcopy0.1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save MYMODELV8_beforeblock_dropcopy0.1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save MYMODELV8_beforeblockatten_dropcopy0.10.1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV1_DW_6BLOCK_DROP --save MYMODELV1_beforeblockatten_dropcopy0.10.1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save v8_x3_skip_32bsize_tiny --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save v8_x3_skip_32bsize --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save V8_x3_skip_32bsize_DF2K_pre --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_x3_skip_32bsize_DF2K/model/model_best.pt --scale 3 --lr 3e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=800
#python main.py --model MYMODELV8 --save V8_x4_skip_32bsize_DF2K_pre --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_x4_skip_32bsize_DF2K/model/model_best.pt --scale 4 --lr 3.75e-5 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=200
#python main.py --model MYMODELV1_DW_6BLOCK --save test_only1 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --n_feats 48 --test_only --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/mymodel_v1_dwUPA_6b_x4/model/model_best.pt
#python main.py --model MYMODELV8_drop --save MYMODELV8_beforeblockatten_dropcopy0.1_right_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8 --save v8_x3_skip_32bsize_DF2K_large --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 60 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8_DROP --save MYMODELV8_beforeblockatten_dropcopy0.1_right_x2_continue --scale 2 --lr 3e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=800 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_beforeblockatten_dropcopy0.1_right_x2/model/model_best.pt
#python main.py --model MYMODELV8 --save v8_x3_skip_32bsize_DF2K_large_continue --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_x3_skip_32bsize_DF2K_large/model/model_best.pt --scale 3 --lr 7.5e-5 --batch_size 32 --patch_size 192 --n_feats 60 --decay 100 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=300
#python main.py --model MYMODELV8_NEWUPSAMPLE --save v8_newupsample_x4_Div2K --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#HF_Ablation
#python main.py --model MYMODELV8_ABLATION_HFENHANCE --save v8_Ablation_noEnhance_x2_Div2K --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_HFENHANCE --save v8_Ablation_2conv1_x2_Div2K --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#HF_Ablation_tiny
#python main.py --model MYMODELV8_ABLATION_HF3CONV1 --save MYMODELV8_Ablation_hf3conv1_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 400-600-800 --data_test Set5 --resume -1 --load MYMODELV8_Ablation_hf3conv1_tiny_x4 --epoch=1000
#python main.py --model MYMODELV8_ABLATION_NOHF_11BLOCK --save MYMODELV8_Ablation_nohf_11block_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 400-600-800 --data_test Set5 --resume -1 --load MYMODELV8_Ablation_nohf_11block_tiny_x4 --epoch=1000
#python main.py --model MYMODELV8_ABLATION_NOHF --save MYMODELV8_Ablation_nohf_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 400-600-800 --data_test Set5 --resume -1 --load MYMODELV8_Ablation_nohf_tiny_x4 --epoch=1000
#python main.py --model MYMODELV8_ABLATION_HFENHANCE --save v8_Ablation_2conv1_x2_Div2K --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_NOHF_11BLOCK --save MYMODELV8_Ablation_nohf_11block_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_nohf_11block_tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#python main.py --model MYMODELV8_ABLATION_NOHF --save MYMODELV8_Ablation_nohf_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_nohf_tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only


#python main.py --model MYMODELV8_ABLATION_HFNOSCA --save MYMODELV8_Ablation_hfnosca_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_HFNOPA --save MYMODELV8_Ablation_hfnopa_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_NOHF --save MYMODELV8_Ablation_nohf_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_HF1CONV3 --save MYMODELV8_Ablation_hf1conv3_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#newHF_Ablation
#python main.py --model MYMODELV8_NEWHF_ADD_ABLATION_1CONV3 --save v8_newhf_add_ablation_1conv3_x2_Div2K --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_NEWHF_ADD_ABLATION_1CONV3 --save v8_newhf_add_ablation_1conv3_x4_Div2K --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_NEWHF_ADD_ABLATION_11BLOCK --save v8_newhf_add_ablation_11block_x4_Div2K --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#new upsample
#python main.py --model V8_NEWUPSAMPLE_ADD --save v8_newupsample_add_x3_Div2K --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model V8_NEWUPSAMPLE_ADD --save v8_newupsample_add_x4_Div2K --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#new HF
#python main.py --model MYMODELV8_NEWHF_ADD --save v8_newhf_add_x4_Div2K --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#HF_interaction_Ablation_tiny
#python main.py --model MYMODELV8_ABLATION_CASCADE_PA_SCA --save MYMODELV8_Ablation_cascade_pa_sca__tiny_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_CASCADE_SCA_PA --save MYMODELV8_Ablation_cascade_sca_pa_tiny_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_PARALLEL_CAT --save MYMODELV8_Ablation_parallel_cat_tiny_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_CASCADE_PA_SCA --save MYMODELV8_Ablation_cascade_pa_sca__tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_CASCADE_SCA_PA --save MYMODELV8_Ablation_cascade_sca_pa_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_PARALLEL_CAT --save MYMODELV8_Ablation_parallel_cat_tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model MYMODELV8_ABLATION_CASCADE_PA_SCA --save MYMODELV8_Ablation_cascade_pa_sca__tiny_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 600-800 --data_test Set5 --resume -1 --load MYMODELV8_Ablation_cascade_pa_sca__tiny_x4 --epoch=1000

#python main.py --model MYMODELV8_ABLATION_CASCADE_SCA_PA --save ./A_visual/MYMODELV8_Ablation_cascade_sca_pa_tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_cascade_sca_pa_tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt


#python main.py --model MYMODELV8_ABLATION_PARALLEL_CAT --save MYMODELV8_Ablation_parallel_cat_tiny_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_parallel_cat_tiny_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_ABLATION_CASCADE_PA_SCA --save MYMODELV8_Ablation_cascade_pa_sca__tiny_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_cascade_pa_sca__tiny_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only



###MYMODEL test_only
#python main.py --model MYMODELV8_ABLATION_HF1CONV3 --save MYMODELV8_Ablation_hf1conv3_x2 --scale 2 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_hf1conv3_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_ABLATION_NOHF --save MYMODELV8_Ablation_nohf_x2 --scale 2 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_nohf_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_ABLATION_HFNOSCA --save MYMODELV8_Ablation_hfnosca_x2 --scale 2 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_hfnosca_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_ABLATION_HFNOPA --save MYMODELV8_Ablation_hfnopa_x2 --scale 2 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_hfnopa_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#python main.py --model MYMODELV8_NEWHF_ADD_ABLATION_1CONV3 --save v8_newhf_add_ablation_1conv3_x4_Div2K --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_newhf_add_ablation_1conv3_x4_Div2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

#python main.py --model MYMODELV8_NEWHF_ADD_ABLATION_11BLOCK --save v8_newhf_add_ablation_11block_x4_Div2K --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_newhf_add_ablation_11block_x4_Div2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_ABLATION_HF1CONV1 --save MYMODELV8_Ablation_hf1conv1_x4 --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/MYMODELV8_Ablation_hf1conv1_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_NEWHF_ADD --save v8_newhf_add_x4_Div2K --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_newhf_add_x4_Div2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model V8_NEWUPSAMPLE_ADD --save v8_newupsample_add_x4_Div2K --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_newupsample_add_x4_Div2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model V8_NEWUPSAMPLE_ADD --save v8_newupsample_add_x3_Div2K --scale 3 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_newupsample_add_x3_Div2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8 --save V8_x4_skip_32bsize_DF2K_pre_test --scale 4 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_x4_skip_32bsize_DF2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model MYMODELV8_ADDDEEP --save_results --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --batch_size 32 --patch_size 128 --n_feats 48 --test_only --pre_train /home/tyh123456/PycharmProject/SRProject_38/experiment/MYMODELV8_Ablation_10block_Div2K_x2/model/model_best.pt
#python main.py --model MYMODELV8_ABLATION_HFENHANCE --save v8_Ablation_noEnhance_x2_Div2K_test --scale 2 --n_feats 48 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/v8_Ablation_noEnhance_x2_Div2K/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only


#####newmodel_v2_noshare_x2 train
#python main.py --model NEWMODEL_V2 --save newmodelv2_x2_48C_117G --scale 2 --lr 6e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv1_dw_noshare_x3 train
#python main.py --model MYMODELV1_DW_6BLOCK --save mymodel_v1_dwUPA_6b_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv6_noshare_x4 train
#python main.py --model MYMODELV1_DW_6BLOCK --save mymodel_v1_dwUPA_6b_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000

#####MYMODELv1_dw_noshare_drop_x2 train
#######python main.py --model MYMODELV1_DW_6BLOCK_DROP --save MYMODELV1_beforeblockatten_dropcopy0.10.05_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#python main.py --model MYMODELV1_DW_6BLOCK --save mymodel_v1_dwUPA_6b_16bsize_x2 --scale 2 --lr 6e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv1_dw_noshare_x3 train
#python main.py --model MYMODELV1_DW_6BLOCK --save MYMODELV1_dw_6BLOCK_pre_x3 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/mymodel_v1_dw_noshare_x3_48C_6e/model/model_best.pt --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####MYMODELv6_share_x4 train
#python main.py --model MYMODELV1_DW_6BLOCK --save mymodel_v1_dw_noshare_x4_48C_6e --scale 4 --lr 6e-4 --batch_size 16 --patch_size 256 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000


#####MYMODELv1_ud_noshare_6b_x3 train
#python main.py --model MYMODELV1_UD_6BLOCK --save mymodel_v1_ud_noshare_x3_48C_81G --scale 3 --lr 6e-4 --batch_size 16 --patch_size 192 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####PAN_x2 train
#python main.py --model PAN --save PAN_X2 --scale 2 --lr 1e-3 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####PAN_drop_x2 train
#python main.py --model PAN_DROP --save pan_x2_lrelu_conv_1_0.05 --scale 2 --lr 1e-3 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####V8_x2 train
#python main.py --model MYMODELV8 --save v8_x2_skip_32bsize --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####V8_drop_x2 train
#python main.py --model MYMODELV8 --save test3 --scale 2 --lr 6e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000
#####RFDN_drop_x2 train
#python main.py --model RFDN --save rfdn_x2_drop0.1_beforeblock_copy --scale 2 --lr 5e-4 --batch_size 16 --patch_size 128 --n_feats 48 --decay 200-400-600-800 --data_test Set5+Set14+B100+Urban100+Manga109 --reset --epoch=1000

#####MYMODEL only_test
#python main.py --model MYMODELV1_DW_6BLOCK --save test9 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --test_only --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/mymodel_v1_dwUPA_6b_32bsize_x2/model/model_latest.pt




# EDSR baseline model (x3) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 3 --patch_size 144 --save edsr_baseline_x3 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR baseline model (x4) - from EDSR baseline model (x2)
#python main.py --model EDSR --scale 4 --save edsr_baseline_x4 --reset --pre_train [pre-trained EDSR_baseline_x2 model dir]

# EDSR in the paper (x2)
#python main.py --model EDSR --scale 2 --save edsr_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset

# EDSR in the paper (x3) - from EDSR (x2)
#python main.py --model EDSR --scale 3 --save edsr_x3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR model dir]

# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]

# MDSR baseline model
#python main.py --template MDSR --model MDSR --scale 2+3+4 --save MDSR_baseline --reset --save_models

# MDSR in the paper
#python main.py --template MDSR --model MDSR --scale 2+3+4 --n_resblocks 80 --save MDSR --reset --save_models

# Standard benchmarks (Ex. EDSR_baseline_x4)
#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble

#python main.py --data_test Set5+Set14+B100+Urban100+DIV2K --data_range 801-900 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train download --test_only --self_ensemble

# Test your own images
#python main.py --model MYEDSR --scale 2 --test_only --data_test DIV2K --data_range 801-900 --pre_train ../experiment/myedsr_SE_x2/model/model_best.pt --save_results

# Advanced - Test with JPEG images
#python main.py --model MDSR --data_test Demo --scale 2+3+4 --pre_train download --test_only --save_results

# Advanced - Training with adversarial loss
#python main.py --template GAN --scale 4 --save edsr_gan --reset --patch_size 96 --loss 5*VGG54+0.15*GAN --pre_train download

# RDN BI model (x2)
#python3.6 main.py --scale 2 --save RDN_D16C8G64_BIx2 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 64 --reset
# RDN BI model (x3)
#python3.6 main.py --scale 3 --save RDN_D16C8G64_BIx3 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 96 --reset
# RDN BI model (x4)
#python3.6 main.py --scale 4 --save RDN_D16C8G64_BIx4 --model RDN --epochs 200 --batch_size 16 --data_range 801-805 --patch_size 128 --reset

# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
# pretrained model can be downloaded from https://www.dropbox.com/s/mjbcqkd4nwhr6nu/models_ECCV2018RCAN.zip?dl=0
#python main.py --template RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --reset --save_results --patch_size 96
# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
#python main.py --template RCAN --save RCAN_BIX3_G10R20P48 --scale 3 --reset --save_results --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX4_G10R20P48, input=48x48, output=192x192
#python main.py --template RCAN --save RCAN_BIX4_G10R20P48 --scale 4 --reset --save_results --patch_size 192 --pre_train ../experiment/model/RCAN_BIX2.pt
# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#python main.py --template RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --reset --save_results --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt

#######################################################################################################################
#####work2
#python main.py --model NET_V22 --save NET_V22_lr_x2 --scale 2 --lr 2e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V23 --save NET_V23_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V31 --save NET_V31_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V32 --save NET_V32_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V32 --save ./A_work2/NET_V32_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V33 --save NET_V33_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V33 --save NET_V33_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V35 --save NET_V35_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V35 --save NET_V35_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V36 --save NET_V36_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V38 --save NET_V38_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V38 --save NET_V38_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V41 --save NET_V41_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V41 --save NET_V41_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_12_5 --save ./A_work2/NET_V39_12_5_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_12_5 --save ./A_work2/NET_V39_12_5_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_12_5 --save ./A_work2/NET_V39_12_5_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_12_5_1 --save ./A_work2/NET_V39_12_5_1_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_12_5_1 --save ./A_work2/NET_V39_12_5_1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13 --save ./A_work2/NET_V39_13_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_1 --save ./A_work2/NET_V39_13_1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_2 --save ./A_work2/NET_V39_13_2_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_2 --save ./A_work2/NET_V39_13_2_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_6 --save ./A_work2/NET_V39_13_6_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_6 --save ./A_work2/NET_V39_13_6_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_6 --save ./A_work2/NET_V39_13_6_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_7 --save ./A_work2/NET_V39_13_7_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_7 --save ./A_work2/NET_V39_13_7_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_7_1 --save ./A_work2/NET_V39_13_7_1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_7_1 --save ./A_work2/NET_V39_13_7_1_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_7_1 --save ./A_work2/NET_V39_13_7_1_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_8_3 --save ./A_work2/NET_V39_13_8_3_x4 --scale 4 --lr 6e-4 --batch_size 32 --patch_size 256 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_10_1 --save ./A_work2/NET_V39_13_10_1_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 38 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_8_4 --save ./A_work2/NET_V39_13_8_4_x2 --scale 2 --lr 6e-4 --batch_size 32 --patch_size 128 --n_feats 38 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_12 --save ./A_work2/NET_V39_13_12_x2 --scale 2 --n_feats 34 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_13_12 --save ./A_work2/NET_V39_13_12_x4 --scale 4 --n_feats 34 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_1_6b --save ./A_work2/NET_V39_14_1_6b_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_1_7b --save ./A_work2/NET_V39_14_1_7b_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_1_9b --save ./A_work2/NET_V39_14_1_9b_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_1_10b --save ./A_work2/NET_V39_14_1_10b_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_6 --save ./A_work2/NET_V39_14_6_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_6 --save ./A_work2/NET_V39_14_6_x3 --scale 3 --lr 6e-4 --batch_size 32 --patch_size 192 --n_feats 32 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_6 --save ./A_work2/NET_V39_14_6_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_7 --save ./A_work2/NET_V39_14_7_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_7 --save ./A_work2/NET_V39_14_7_x3 --scale 3 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 192 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_7 --save ./A_work2/NET_V39_14_7_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8 --save ./A_work2/NET_V39_14_8_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8_1 --save ./A_work2/NET_V39_14_8_1_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_9 --save ./A_work2/NET_V39_14_9_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8 --save ./A_work2/NET_V39_14_8_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_9 --save ./A_work2/NET_V39_14_9_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8_2 --save ./A_work2/NET_V39_14_8_2_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8 --save ./A_work2/NET_V39_14_8_x3 --scale 3 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 192 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8_2 --save ./A_work2/NET_V39_14_8_2_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8_2 --save ./A_work2/NET_V39_14_8_2_x3 --scale 3 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 192 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8_LARGE --save ./A_work2/NET_V39_14_8_large_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model NET_V39_14_8_LARGE --save ./A_work2/NET_V39_14_8_large_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#python main.py --model NET_V39_14_8_2 --save ./A_work2/NET_V39_14_8_2_2_x4 --scale 3 --n_feats 64 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

##ablation_erab
#python main.py --model ERAB_ONLY_SOBELX --save ./A_work2/erab_only_sobelx_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

##ablation_up
#python main.py --model UP_UPA --save ./A_work2/up_upa_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_UPA --save ./A_work2/up_upa_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_IF_I0 --save ./A_work2/up_if_i0_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_IF_I0 --save ./A_work2/up_if_i0_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_IM_I0 --save ./A_work2/up_im_i0_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_IM_I0 --save ./A_work2/up_im_i0_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_IM --save ./A_work2/up_im_x2 --scale 2 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model UP_IM --save ./A_work2/up_im_x4 --scale 4 --n_feats 32 --lr 6e-4 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

###test
#python main.py --model NET_V39_14_8_2 --save ./A_work2/NET_V39_14_8_2_x2 --scale 2 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/A_work2/NET_V39_14_8_2_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_gt
#python main.py --model NET_V39_14_8_2 --save ./A_work2/NET_V39_14_8_2_x3 --scale 3 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/A_work2/NET_V39_14_8_2_x3/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model UP_IM --save ./A_work2/up_im_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/A_work2/up_im_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model NET_V32 --save ./A_work2/NET_V32_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/A_work2/NET_V32_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

###continue
#python main.py --model NET_V39_2 --save ./A_work2/NET_V39_2_x2 --scale 2 --lr 6e-4 --patch_size 128 --batch_size 32 --n_feats 32 --decay 1000 --data_test Set5 --epoch=1000 --resume -1 --load A_work2/NET_V39_2_x2
#python main.py --model NET_V39_12_8 --save ./A_work2/NET_V39_12_8_x4 --scale 4 --lr 6e-4 --patch_size 256 --batch_size 32 --n_feats 32 --decay 600-800 --data_test Set5 --epoch=1000 --resume -1 --load A_work2/NET_V39_12_8_x4


###vision_test
#python main.py --model NET_V39_14_8_2 --save ./A_visual/NET_V39_14_8_2_x2 --scale 2 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/NET_V39_14_8_2_x2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model NET_V39_14_8_2 --save ./A_visual/NET_V39_14_8_2_x3 --scale 3 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/NET_V39_14_8_2_x3/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model NET_V39_14_8_2 --save ./A_visual/NET_V39_14_8_2_x4 --scale 4 --n_feats 32 --pre_train /home/tyh123456/PycharmProjects/SRProjects_38/experiment/A_work2/NET_V39_14_8_2_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model shufflemixer --save ./A_visual/shufflemixer_x4_results --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/shufflemixer_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model shufflemixer --save ./A_visual/shufflemixer_x4_results --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/shufflemixer_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model scnet --save ./A_visual/scnet_x4_results --scale 4 --n_feats 64 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/scnet/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt --rgb_range 255
#python main.py --model man_arch_mbf_u_4_6xmab --save ./A_visual/man_x4_results --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_arch_mbf_u_4_6xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model man_arch --save ./A_visual/man_x2_results --scale 2 --n_feats 36 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/pre_models/MAN-tiny/MAN-Tiny-x2.pth --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt --rgb_range 1
#python main.py --model safmn --save ./A_visual/safmn_x4_results --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SAFMN/NTIRE2023_ESR/model_zoo/team15_SAFMN.pth --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt --rgb_range 1
#python main.py --model safmn --save ./A_visual/safmn_x4_results --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SAFMN/SAFMN_NTIRE_ESR_x4/SAFMN_NTIRE_ESR_x4.pth --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model lapar --save ./A_visual/lapar_c_x4_results --scale 4 --n_feats 16 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/lapar_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt
#python main.py --model awsrn --save ./A_visual/arwsrn_x2_results --scale 2 --n_feats 16 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/pre_models/pretrainmodels/AWSRN_Sx2/model/model_latest.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only --save_results --save_gt

### RealSRV3 Visual
#python main.py --model shufflemixer --save ./A_visual/shufflemixer_x4_results_realsrv3 --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/shufflemixer_x4/model/model_best.pt --data_test Realsrv3 --test_only --save_results --save_gt
#python main.py --model scnet --save ./A_visual/scnet_x4_results_realsrv3 --scale 4 --n_feats 64 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/scnet/model/model_best.pt --data_test Realsrv3 --test_only --save_results --save_gt
#python main.py --model man_arch_mbf_u_4_6xmab --save ./A_visual/man_x4_results_realsrv3 --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_arch_mbf_u_4_6xmab/model/model_best.pt --data_test Realsrv3 --test_only --save_results --save_gt
#python main.py --model safmn --save ./A_visual/safmn_x4_results_realsrv3 --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SAFMN/NTIRE2023_ESR/model_zoo/team15_SAFMN.pth --data_test Realsrv3 --test_only --save_results --save_gt --rgb_range 1
#python main.py --model safmn --save ./A_visual/safmn_x4_results_realsrv3 --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SAFMN/SAFMN_NTIRE_ESR_x4/SAFMN_NTIRE_ESR_x4.pth --data_test Realsrv3 --test_only --save_results --save_gt
#python main.py --model lapar --save ./A_visual/lapar_c_x4_results_realsrv3 --scale 4 --n_feats 16 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/lapar_x4/model/model_best.pt --data_test Realsrv3 --test_only --save_results --save_gt
#python main.py --model NET_V39_14_8_2 --save ./A_visual/erdrn_x4_results_realsrv3 --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/NET_V39_14_8_2_x4/model/model_best.pt --data_test Realsrv3 --test_only --save_results --save_gt


###man_arch train
#python main.py --model man_arch --save ./A_work2/MAN_0_x4 --scale 4 --n_feats 48 --lr 5e-4  --rgb_range 1 --batch_size 32 --patch_size 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_arch --save ./A_work2/man_arch_1_x4 --scale 4 --n_feats 48 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
##test
#python main.py --model man_arch --save man_arch_test_1 --scale 4 --n_feats 48 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_arch_1_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only


######  MBF_U test in other network

###man_arch_MBF_U_1_X4 train
#python main.py --model man_arch_mbf_u_1 --save ./A_work2/man_arch_mbf_u_1 --scale 4 --n_feats 48 --lr 5e-4  --rgb_range 1 --batch_size 32 --patch_size 48 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_arch_mbf_u_1 --save ./A_work2/man_arch_mbf_u_1 --scale 4 --n_feats 48 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set14 --reset --epoch=1000
##text
#python main.py --model man_arch_mbf_u_1 --save man_arch_mbf_u_1_test_1 --scale 4 --n_feats 48 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_arch_mbf_u_1/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only


###man_arch_MBF_U_2_4xmab_X4 train
#python main.py --model man_arch_mbf_u_2_4xmab --save ./A_work2/man_arch_mbf_u_2_4xmab --scale 4 --n_feats 48 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_arch_mbf_u_3_4xmab --save ./A_work2/man_arch_mbf_u_3_4xmab --scale 4 --n_feats 36 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_arch_mbf_u_4_6xmab --save ./A_work2/man_arch_mbf_u_4_6xmab --scale 4 --n_feats 36 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_mbf_u_5_5xmab --save ./A_work2/man_mbf_u_5_5xmab --scale 4 --n_feats 30 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_mbf_u_6_5xmab --save ./A_work2/man_mbf_u_6_5xmab --scale 4 --n_feats 36 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model man_mbf_u_6_5xmab --save ./A_work2/man_mbf_u_6_5xmab_2 --loss 1*L1+1*FFT --scale 4 --n_feats 36 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

##test
#python main.py --model man_arch_mbf_u_2_4xmab --save man_arch_mbf_u_2_test_1 --scale 4 --n_feats 48 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_arch_mbf_u_2_4xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model man_arch_mbf_u_4_6xmab --save man_arch_mbf_u_4_test_1 --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_arch_mbf_u_4_6xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model man_mbf_u_5_5xmab --save man_arch_mbf_u_4_test_2 --scale 4 --n_feats 30 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_mbf_u_5_5xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model man_mbf_u_6_5xmab --save man_arch_mbf_u_6_test_1 --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_mbf_u_6_5xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model man_mbf_u_6_5xmab --save man_arch_mbf_u_6_test_1 --scale 4 --n_feats 36 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_mbf_u_6_5xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

###re-train
#python main.py --model man_arch_mbf_u_2_4xmab --save ./A_work2/man_mbf_u_2_4xmab --scale 4 --n_feats 48 --lr 6e-4  --rgb_range 255 --batch_size 32 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

#test
#python main.py --model man_arch_mbf_u_2_4xmab --save man_mbf_u_2_test_1 --scale 4 --n_feats 48 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/man_mbf_u_2_4xmab/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

### shufflemixer
##train
#python main.py --model shufflemixer --save ./A_work2/shufflemixer_x4 --loss 1*L1+0.1*FFT --scale 4 --n_feats 32 --lr 5e-4  --rgb_range 255 --batch_size 64 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model shufflemixer_mbf_u_1 --save ./A_work2/shufflemixer_mbf_u_1 --loss 1*L1+0.1*FFT --scale 4 --n_feats 32 --lr 5e-4  --rgb_range 255 --batch_size 64 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model shufflemixer_mbf_u_2 --save ./A_work2/shufflemixer_mbf_u_2 --loss 1*L1+0.1*FFT --scale 4 --n_feats 24 --lr 5e-4  --rgb_range 255 --batch_size 64 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

##test
#python main.py --model shufflemixer --save shufflemixer_x4_test --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/shufflemixer_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model shufflemixer_mbf_u_1 --save shufflemixer_mbf_u_1_test --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/shufflemixer_mbf_u_1/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model shufflemixer_mbf_u_2 --save shufflemixer_mbf_u_2_test --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/shufflemixer_mbf_u_2/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only

### scnet
##train
#python main.py --model scnet --save ./A_work2/scnet --loss 1*L1 --scale 4 --n_feats 64 --lr 2e-4  --rgb_range 255 --batch_size 32 --seed 0 --patch_size 256 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#CUDA_VISIBLE_DEVICES=2 python basicsr/train.py -opt options/train/SCNet/scnet_mbf-x4.yml
#CUDA_VISIBLE_DEVICES=3 python basicsr/train.py -opt options/train/SCNet/scnet_mbf_2-x4.yml

##test
#python main.py --model scnet --save scnet_test --scale 4 --n_feats 64 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/scnet/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only


### lapar
##train
#python main.py --model lapar --save ./A_work2/lapar_x4 --loss 1*L1 --scale 4 --n_feats 32 --lr 4e-4  --rgb_range 255 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model lapar_mbf_1 --save ./A_work2/lapar_mbf_1_x4 --loss 1*L1 --scale 4 --n_feats 32 --lr 4e-4  --rgb_range 255 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model lapar_mbf_2 --save ./A_work2/lapar_mbf_2_x4 --loss 1*L1 --scale 4 --n_feats 32 --lr 4e-4  --rgb_range 255 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000
#python main.py --model lapar_mbf_3 --save ./A_work2/lapar_mbf_3_x4 --loss 1*L1 --scale 4 --n_feats 32 --lr 4e-4  --rgb_range 255 --batch_size 32 --patch_size 128 --decay 200-400-600-800 --data_test Set5 --reset --epoch=1000

##test
#python main.py --model lapar --save lapar_test --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/lapar_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only
#python main.py --model lapar_mbf_2 --save lapar_mbf_2_test --scale 4 --n_feats 32 --pre_train /home/xingrenwang2024/SRProjects_work2/SRProjects_38/experiment/A_work2/lapar_mbf_2_x4/model/model_best.pt --data_test Set5+Set14+B100+Urban100+Manga109 --test_only


