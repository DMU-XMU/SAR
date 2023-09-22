# # DMC setting
# python launch.py \
# --env dmc.finger.spin \
# --agent drq --base sac \
# --auxiliary cresp \
# --num_sources 2 \
# --dynamic -bg -tbg \
# --disenable_default \
# --critic_lr 5e-4 \
# --actor_lr 5e-4 \
# --alpha_lr 5e-4 \
# --extr_lr 5e-4 \
# --nstep_of_rsd 5 \
# --num_sample 256 \
# --opt_mode max \
# --omega_opt_mode min_mu \
# --rs_fc \
# --discount_of_rs 0.8 \
# --extr_update_via_qfloss True \
# --cuda_id 0

# Carla setting 
export PYTHONPATH="/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg"
python launch.py \
--env dmc.cheetah.run \
--agent curl --base sac \
--auxiliary sar \
--num_sources 2 \
--dynamic -bg -tbg \
--disenable_default \
--critic_lr 5e-4 \
--actor_lr 5e-4 \
--alpha_lr 5e-4 \
--extr_lr 5e-4 \
--targ_extr 1 \
--nstep_of_rsd 5 \
--num_sample 128 \
--batch_size 128 \
--opt_mode max \
--omega_opt_mode min_mu \
--rs_fc \
--discount_of_rs 0.8 \
--extr_update_via_qfloss True \
--cuda_id 0 \
--port 4021 \
--exp_name sar \
-s 0 1 2 3 4
