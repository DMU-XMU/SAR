import copy
from module.extr_module import make_extr


def init_extractor(obs_shape, env, device, extr_config):
    extr = make_extr(obs_shape=obs_shape, env=env, **extr_config).to(device)
    extr_targ = None
    if extr_config['targ_extr']:
        extr_targ = copy.deepcopy(extr)
    extr_tau = extr_config['extr_tau']
    out_dict = dict(extr=extr, extr_targ=extr_targ, extr_tau=extr_tau)
    return out_dict


def init_algo(ALGO, repr_dict, algo_config):
    return ALGO(repr_dict=repr_dict, **algo_config)


def init_auxiliary_task(AUXLILIARY, action_shape, aux_config, device):
    return None if AUXLILIARY is None \
        else AUXLILIARY(action_shape=action_shape, device=device, **aux_config)

from augmentation import *

_AVAILABLE_AUG_FUNC = {
    'shift_car': RandomShiftsAug_420,
    'shift_dmc': RandomShiftsAug_84
}


def init_aug_func(afunc, env, image_pad=None):
    assert afunc in 'shift'
    if "carla" in env:
        aug_funcs = _AVAILABLE_AUG_FUNC['shift_car'](image_pad) if image_pad is not None \
                else _AVAILABLE_AUG_FUNC['shift_car']()
        #aug_funcs = aug_shift if afunc == 'shift_car' else _AVAILABLE_AUG_FUNC[afunc]()
    elif "dmc" in env:
        aug_funcs = _AVAILABLE_AUG_FUNC['shift_dmc'](image_pad) if image_pad is not None \
                else _AVAILABLE_AUG_FUNC['shift_dmc']()
        #aug_funcs = aug_shift if afunc == 'shift_dmc' else _AVAILABLE_AUG_FUNC[afunc]()        
    return aug_funcs


def init_agent(AGENT, config):
    aug_funcs = None if config['evaluation'] \
        else init_aug_func(config['data_aug'], config['env'], config['agent_params']['image_pad'])
    return AGENT(aug_funcs, config=config, **config['agent_params'])
