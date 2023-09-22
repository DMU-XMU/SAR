import time
import numpy as np
from collections import deque

import common.utils as utils

def evaluate(test_env, agent, logger, video, num_eval_episodes, start_time, action_repeat, num_episode, step):

    def eval(num, env, agent, logger, video, num_episodes, step, start_time, action_repeat, num_episode):
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
            'Episode': num_episode,
            'Time': (time.time() - start_time) / 3600,
        }

        utils.log(logger, 'eval_agent', test_info, step)
        #video.save('%d.mp4' % step)


    if isinstance(test_env, list):
        # test_env is a list
        for num, t_env in enumerate(test_env):
            eval(num, t_env, agent, logger, video, num_eval_episodes,
                 step, start_time, action_repeat, num_episode)
    else:
        # test_env is an environment
        eval(0, test_env, agent, logger, video, num_eval_episodes,
             step, start_time, action_repeat, num_episode)



def train_agent(train_envs, test_env, agent, replay_buffer, logger, video, model_dir,
                total_steps, init_steps, eval_freq, action_repeat, num_updates,
                num_eval_episodes, test, save_model, save_model_freq, device, **kwargs):

    epoch_start_times = start_time = time.time()
    env_id, num_sources, best_return = 0, len(train_envs), 0.0
    episode, episode_reward, episode_step, done = 0, 0, 0, False
    env = train_envs[env_id]
    o = env.reset() #(9,,)
    replay_buffer.add_obs(o)
    # import pdb
    # pdb.set_trace()

    for step in range(1, total_steps + 1):

        # sample action for data collection
        if step < init_steps:
            a = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                a = agent.select_action(o)

        # run training update
        if step >= init_steps and step % num_updates == 0:
            for _ in range(num_updates):
                agent.update(replay_buffer, logger, step, step % 500 == 0)

        o2, r, done, infos = env.step(a)

        agent.total_time_steps += 1
        episode_reward += r
        episode_step += 1
        # allow infinit bootstrap
        d_bool = 0 if episode_step == env._max_episode_steps else float(done)
        replay_buffer.add(o, a, r, o2, done, d_bool, episode_step, env_id)

        o = o2

        if done:
            train_info = dict(EpRet=episode_reward, EpLen=episode_step, EpNum=episode)
            utils.log(logger, 'train_agent', train_info, step)
            # logger['tb'].dump(step)
            print("Total T: {} Reward: {:.3f} Episode Num: {} Episode T: {} Time: {}".format(
                agent.total_time_steps, episode_reward, episode, episode_step, utils.calc_time(start_time)))
            if best_return < episode_reward:
                best_return = episode_reward
                agent.save(model_dir, 'best_in_env_%d' % env_id)

            episode += 1
            env_id = episode % num_sources
            env = train_envs[env_id]
            o, done, episode_reward, episode_step = env.reset(), False, 0, 0
            replay_buffer.add_obs(o)

        # evaluate agent periodically
        if step % eval_freq == 0 and step > init_steps and test:

            evaluate(test_env, agent, logger, video, num_eval_episodes,
                     start_time, action_repeat, episode, step)

            print('Update Extr Times: %s Update Critic Times: %d Update Actor Times: %d' % (
                agent.update_extr_total_steps, agent.update_critic_steps, agent.update_actor_steps
            ))
            agent.print_log(logger['sp'],
                            test_env,
                            step // eval_freq,
                            step,
                            action_repeat,
                            test,
                            start_time,
                            eval_freq * action_repeat / (time.time() - epoch_start_times))

            if save_model and step // eval_freq == save_model_freq:
                agent.save(model_dir, step)
