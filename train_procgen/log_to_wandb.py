import os
import csv
import tqdm
import wandb

PPO_PATH = '/raid0/dian/procgen_baseline'
INPUT = 'rgb'
METHOD = f'PPO_{INPUT}'

def main(config):
    
    wandb.init(project=config.project, config={
        'method': METHOD,
        'env_name': config.envname,
        'num_levels': 200,
        'seed': config.seed,
    })
    
    with open(os.path.join(PPO_PATH, config.envname, f'ppo_{INPUT}_{config.seed}', 'progress.csv'), 'r') as f:
    # with open(os.path.join(PPO_PATH, config.envname, f'ppo_run{config.seed}', 'progress.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm.tqdm(reader):
            N = int(row['misc/total_timesteps'])
            train_eprets = float(row['eprewmean'])
            eval_eprets = float(row['eval_eprewmean'])
            
            wandb.log({
                'num_frames': N,
                'train_eprets': train_eprets,
                'eval_eprets': eval_eprets,
            })



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname')
    parser.add_argument('seed', type=int)
    parser.add_argument('--project', default='procgen_ours5')
    
    config = parser.parse_args()
    main(config)