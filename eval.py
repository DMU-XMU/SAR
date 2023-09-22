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

_AVAILABLE_AGENT = {'drq': DrQAgent, 'curl': CurlAgent}
_AVAILABLE_AUXILIARY = {'cresp': CRESP, 'sar': SAR}
_AVAILABLE_ALGORITHM = {'sac': SAC, 'td3': TD3}
def Evaluate(test_env, agent, video, num_eval_episodes, start_time, action_repeat, num_episode, step):
    
    def eval(num, env, agent, video, num_episodes, step, start_time, action_repeat, num_episode):
        all_ep_rewards = []
        all_ep_length = []
        # loop num_episodes
        for episode in range(num_episodes):
            obs = env.reset()
            #video.init(enabled=(episode == 0))
            done, episode_reward, episode_length = False, 0, 0
            # evaluate once
            while not done:
                with utils.eval_mode(agent):
                    action = agent.select_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                #video.record(env)
                episode_reward += reward
                episode_length += 1

            video.save('%s-Env%d.mp4' % (step * action_repeat, num))
            # record the score
            all_ep_rewards.append(episode_reward)
            all_ep_length.append(episode_length)

        # record log
        mean, std, best = np.mean(all_ep_rewards), np.std(all_ep_rewards), np.max(all_ep_rewards)
        test_info = {
            ("TestEpRet%d" % num): mean,
            ("TestStd%d" % num): std,
            ("TestBestEpLen%d" % num): best,
            'Episode': num_episode,
            'Time': (time.time() - start_time) / 3600,
        }
        print(test_info)
        #utils.log(logger, 'eval_agent', test_info, step)

    if isinstance(test_env, list):
        # test_env is a list
        for num, t_env in enumerate(test_env):
            eval(num, t_env, agent, video, num_eval_episodes,
                 step, start_time, action_repeat, num_episode)
    else:
        # test_env is an environment
        eval(0, test_env, agent, video, num_eval_episodes,
             step, start_time, action_repeat, num_episode)


def run_eval(args, device, work_dir, config):
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    _, domain_name, task_name = args.env.split('.')

    # Initialize Logger and Save Hyperparameters
    logger, work_dir = utils.init_logger(args, config, work_dir)
    video_dir = utils.make_dir(work_dir / 'video')
    model_dir = utils.make_dir(work_dir / 'model')
    buffer_dir = utils.make_dir(work_dir / 'buffer')
    print(model_dir)
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

    # Initialize Environment
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
    obs_shape, pre_aug_obs_shape = obs_dict
    action_shape = train_envs[0].action_space.shape
    action_limit = train_envs[0].action_space.high[0]

    config.update(dict(obs_shape=obs_shape, batch_size=args.batch_size, device=device))
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
    agent.load(model_dir, step)
    for step in [0,1,2,3]:
        Evaluate(test_env=test_env,
                agent=agent,
                video=video,
                start_time=time.time(),
                num_episode = None,
                step=step,
                num_eval_episodes=2,
                action_repeat=4)

    for env in train_envs:
        env.close()
    test_env.close()


if __name__ == '__main__':

    args = parse_args()
    cuda_id = "cuda:%d" % args.cuda_id
    device = torch.device(cuda_id if args.cuda else "cpu")
    work_dir = Path.cwd()
    config = utils.read_config(args, work_dir / args.config_dir)
    torch.multiprocessing.set_start_method('spawn', force=True)
    args.seed, config['setting']['seed'] = 0, 0
    run_eval(args, device, work_dir, config)
