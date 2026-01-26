python main.py --dataset wn18rr --cuda True --device cuda:0 --batch_size 2048 --max_grad_norm 9.0 --nneg 1024 --npos 1 --margin 1.08 --max_norm 5. --lr 0.05 --gamma 0.9 --step_size 40 --num_epochs 1000 --dim 100 --valid_steps 5 --early_stop 150 --optimizer radam --noise_reg 0.05 --Lorentz_k 0.3 \
    --adv 1.0 --CRR_p 0.0 --CRR_t 4.0 --CRR_w 0.0 --BCE_t 1.5 --BCE_w 1.0 --positive_weight 0.01 --Margin_w 0.0 --run_id 1

python main.py --dataset wn18rr --cuda True --device cuda:0 --batch_size 2048 --max_grad_norm 9.0 --nneg 1024 --npos 1 --margin 1.08 --max_norm 5. --lr 0.05 --gamma 0.9 --step_size 40 --num_epochs 1000 --dim 100 --valid_steps 5 --early_stop 150 --optimizer radam --noise_reg 0.05 --Lorentz_k 0.5 \
    --adv 1.0 --CRR_p 0.0 --CRR_t 4.0 --CRR_w 0.0 --BCE_t 1.5 --BCE_w 1.0 --positive_weight 0.01 --Margin_w 0.0 --run_id 2

python main.py --dataset wn18rr --cuda True --device cuda:0 --batch_size 4096 --max_grad_norm 9.0 --nneg 1024 --npos 1 --margin 1.08 --max_norm 5. --lr 0.05 --gamma 0.9 --step_size 40 --num_epochs 1000 --dim 100 --valid_steps 5 --early_stop 150 --optimizer radam --noise_reg 0.05 --Lorentz_k 0.75 \
    --adv 1.0 --CRR_p 0.0 --CRR_t 4.0 --CRR_w 0.0 --BCE_t 1.5 --BCE_w 1.0 --positive_weight 0.01 --Margin_w 0.0 --run_id 3

python main.py --dataset wn18rr --cuda True --device cuda:0 --batch_size 4096 --max_grad_norm 9.0 --nneg 1024 --npos 1 --margin 1.08 --max_norm 5. --lr 0.05 --gamma 0.9 --step_size 40 --num_epochs 1000 --dim 100 --valid_steps 5 --early_stop 150 --optimizer radam --noise_reg 0.05 --Lorentz_k 1.0 \
    --adv 1.0 --CRR_p 0.0 --CRR_t 4.0 --CRR_w 0.0 --BCE_t 1.5 --BCE_w 1.0 --positive_weight 0.01 --Margin_w 0.0 --run_id 4