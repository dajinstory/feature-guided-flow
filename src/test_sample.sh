# CUDA_VISIBLE_DEVICES=5 python test_sample.py --model baseline
# CUDA_VISIBLE_DEVICES=5 python test_sample.py --model intertemp
# CUDA_VISIBLE_DEVICES=5 python test_sample.py --model recon
# CUDA_VISIBLE_DEVICES=5 python test_sample.py --model featureguidance
CUDA_VISIBLE_DEVICES=3 python test_sample.py --model fg_recon
