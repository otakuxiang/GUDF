for scene in "24" "37" "40" "55" "63" "69" "83" "97" "105" "106" "110" "114" "118" "122" "65"
do 
python train.py -s ../data/dtu/DTU/scan$scene -m output/dtu/scan$scene -r 2 --depth_ratio 0. --multi_view_weight_from_iter 7000 --preload_img --multi_view_ncc_weight 0.5
python render_pgsr.py --iteration 20000 -m output/dtu/scan$scene --depth_ratio 0 
python scripts/eval_dtu/evaluate_single_scene.py --input_mesh output/dtu/scan$scene/mesh/tsdf_fusion_post.ply --scan_id $scene --output_dir output/dtu/scan$scene --mask_dir ../data/dtu/DTU/ --DTU ../data/dtu_eval/
# python scripts/eval_dtu/evaluate_single_scene.py --input_mesh output/dtu/scan$scene/train/ours_30000/fuse_post.ply --scan_id $scene --output_dir output/tmp/scan$scene --mask_dir ../data/dtu/DTU/ --DTU ../data/dtu_eval/
done
