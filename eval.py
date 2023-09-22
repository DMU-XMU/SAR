import torch
import numpy as np
import os
#os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

from common import utils, make_env
from common.buffer_trajectory import BReplayBuffer
# from common.buffer import ReplayBuffer
from common.video import VideoRecorder

from argument import parse_args
from module.init_module import init_agent
from train import train_agent,evaluate

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

from agent import *
from algo import *
from auxiliary import *
from CARLA_.PythonAPI.carla.agents.navigation.carla_env import CarlaEnv

_AVAILABLE_AGENT = {'drq': DrQAgent, 'curl': CurlAgent}
_AVAILABLE_AUXILIARY = {'cresp': CRESP}
_AVAILABLE_ALGORITHM = {'sac': SAC, 'td3': TD3}
def Evaluate(test_env, agent, num_eval_episodes, start_time, action_repeat, num_episode, step):

    def eval(num, env, agent, num_episodes, step, start_time, action_repeat, num_episode):
        all_ep_rewards = []
        all_ep_length = []
        # carla metrics:
        reason_each_episode_ended = []
        distance_driven_each_episode = []
        crash_intensity = 0.
        steer = 0.
        brake = 0.
        count = 0
        do_carla_metrics = True
        # loop num_episodes
        for episode in range(num_episodes):
            obs = env.reset()
            dist_driven_this_episode = 0.
            #video.init(enabled=(episode == 0))
            done, episode_reward, episode_length = False, 0, 0
            # evaluate once
            while not done:
                with utils.eval_mode(agent):
                    action = agent.select_action(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                # metrics:
                if do_carla_metrics:
                    dist_driven_this_episode += info['distance']
                    crash_intensity += info['crash_intensity']
                    steer += abs(info['steer'])
                    brake += info['brake']
                    count += 1
                #video.record(env)

            #video.save('%s-Env%d.mp4' % (step * action_repeat, num))
            # record the score
            all_ep_rewards.append(episode_reward)
            all_ep_length.append(episode_length)

        # record log
        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)

        mean, std, best = np.mean(all_ep_rewards), np.std(all_ep_rewards), np.max(all_ep_rewards)
        if do_carla_metrics:
            print('METRICS--------------------------')
            print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
            print("distance: {}".format(distance_driven_each_episode))
            print('crash_intensity: {}'.format(crash_intensity / num_episodes))
            print('steer: {}'.format(steer / count))
            print('brake: {}'.format(brake / count))
            print('---------------------------------')
        test_info = {
            ("TestEpRet%d" % num): mean,
            ("TestStd%d" % num): std,
            ("TestBestEpLen%d" % num): best,
            "DistanceEp": np.mean(distance_driven_each_episode),
            "Crash_intensity" : crash_intensity / num_episodes,
            "Steer" : steer / count,
            "Brake" : brake / count,
            'EndReason': reason_each_episode_ended,
            'Time': (time.time() - start_time) / 3600,
        }

        with open('./model/sar/data.txt', 'a', encoding='utf-8') as file:
            data_str = str(test_info)
            file.write(data_str)
            file.write('\n')  #

        # file = open('./model/sar/data.txt', 'w')
        # data_str = str(test_info)
        # file.write(data_str)
        #file.close()
        #utils.log(logger, 'eval_agent', test_info, step)
        #video.save('%d.mp4' % step)


    if isinstance(test_env, list):
        # test_env is a list
        for num, t_env in enumerate(test_env):
            eval(num, t_env, agent, num_eval_episodes,
                 step, start_time, action_repeat, num_episode)
    else:
        # test_env is an environment
        eval(0, test_env, agent, num_eval_episodes,
             step, start_time, action_repeat, num_episode)


def run_eval(args, device, work_dir, config):
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    domain, domain_name, task_name = args.env.split('.')

    # Initialize Logger and Save Hyperparameters
    logger, work_dir = utils.init_logger(args, config, work_dir)
    # video_dir = utils.make_dir(work_dir / 'video')
    # model_dir = utils.make_dir(work_dir / 'model')
    # buffer_dir = utils.make_dir(work_dir / 'buffer')
    #print(model_dir)
    #video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    # Initialize Environment
    # train_envs, test_env, obs_dict = make_env.set_dcs_multisources(
    #     domain_name,
    #     task_name,
    #     config['buffer_params']['image_size'],
    #     config['train_params']['action_repeat'],
    #     test_background=args.test_background,
    #     test_camera=args.test_camera,
    #     test_color=args.test_color,
    #     **config['setting']
    # )
    # obs_shape, pre_aug_obs_shape = obs_dict
    # action_shape = train_envs[0].action_space.shape
    # action_limit = train_envs[0].action_space.high[0]
    # Initialize Environment
    if domain == "dmc":
        train_envs, test_env, obs_dict = make_env.set_dcs_multisources(
            domain_name,
            task_name,
            config['buffer_params']['image_size'],
            config['train_params']['action_repeat'],
            test_background=args.test_background,
            test_camera=args.test_camera,
            test_color=args.test_color,
            **config['setting']
        )
    elif domain == "carla":
        print("Runing carla env...")
        env = CarlaEnv(
            render_display=0,#args.render,  # for local debugging only
            display_text=0,#args.render,  # for local debugging only
            record_display_images=0,  # 0, 1
            record_rl_images=0,  # 0, 1
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=config['buffer_params']['image_size'],
            max_episode_steps=1000,
            frame_skip=config['train_params']['action_repeat'],
            is_other_cars=True,
            port= args.port
        )
        env = utils.FrameStack(env, k=args.frame_stack)
        train_envs = []
        train_envs.append(env)
        test_env = train_envs

    #obs_shape, pre_aug_obs_shape = obs_dict
    #print(obs_shape, pre_aug_obs_shape)
    action_shape = train_envs[0].action_space.shape
    action_limit = 1.0 #train_envs[0].action_space.high[0]

    config.update(dict(obs_shape=(9, 84, 420), batch_size=args.batch_size, device=device))
    config['algo_params'].update(dict(action_shape=action_shape,
                                      action_limit=action_limit,
                                      device=device))
    # Initialize Agent
    assert args.agent in _AVAILABLE_AGENT
    config['aux_task'] = None
    if args.auxiliary is not None:
        assert args.auxiliary in _AVAILABLE_AUXILIARY
        config['aux_task'] = _AVAILABLE_AUXILIARY[args.auxiliary]
    assert args.base in _AVAILABLE_ALGORITHM
    config['base'] = _AVAILABLE_ALGORITHM[args.base]
    agent = init_agent(_AVAILABLE_AGENT[args.agent], config)
    agent.load("./model/sar", "best_in_env_0")  #model_dir, 'best_in_env_%d'
    for step in range(100):
        Evaluate(test_env=test_env,
                agent=agent,
                #video=video,
                start_time=time.time(),
                num_episode = None,
                step=step,
                num_eval_episodes=1,
                action_repeat=4)
    # for env in train_envs:
    #     env.close()
    # test_env.close()


if __name__ == '__main__':

    args = parse_args()
    cuda_id = "cuda:%d" % args.cuda_id
    device = torch.device(cuda_id if args.cuda else "cpu")
    work_dir = Path.cwd()
    config = utils.read_config(args, work_dir / args.config_dir)
    torch.multiprocessing.set_start_method('spawn', force=True)
    args.seed, config['setting']['seed'] = 0, 0
    run_eval(args, device, work_dir, config)
