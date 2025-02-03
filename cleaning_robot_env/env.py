import gym
from gym import spaces
import numpy as np

class CleaningRobotEnv(gym.Env):
    def __init__(self, grid_size=2, init_dirt_count=1):
        super(CleaningRobotEnv, self).__init__()
        
        self.num_hits_total = 2
        self.grid_size = grid_size
        self.init_dirt_count = init_dirt_count
        
        # Actions: Up, Down, Left, Right, Clean, Do Nothing
        self.action_space = spaces.Discrete(6)
        
        # New observation space: [robot_x, robot_y, hits_remaining] + flattened dirt grid
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3 + grid_size**2,) if False else (1 + 2 * grid_size**2,),  # 3 for position/hits, rest for dirt
            dtype=np.float32
        )
        
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.reset()
    
    def _get_state(self):
        if False:
            # Normalized features
            robot_x = self.robot_pos[0] / (self.grid_size - 1)
            robot_y = self.robot_pos[1] / (self.grid_size - 1)
            hits_remaining = (self.num_hits_remaining - 1) / (self.num_hits_total - 1)
            
            # Flatten and normalize dirt grid
            dirt_grid = self.grid.flatten().astype(np.float32)
            
            return np.concatenate([
                [robot_x, robot_y, hits_remaining],
                dirt_grid
            ])
        
        # Normalized features
        robot_x = self.robot_pos[0] / (self.grid_size - 1)
        robot_y = self.robot_pos[1] / (self.grid_size - 1)
        hits_remaining = (self.num_hits_remaining - 1) / (self.num_hits_total - 1)
        
        agent_pos_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        agent_pos_grid[self.robot_pos[0], self.robot_pos[1]] = 1
        agent_pos_grid = agent_pos_grid.flatten()

        # Flatten and normalize dirt grid
        dirt_grid = self.grid.flatten().astype(np.float32)
        
        return np.concatenate([
            [hits_remaining],
            agent_pos_grid,
            dirt_grid
        ])
        
    
    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.num_hits_remaining = self.num_hits_total
        
        # Place initial dirt
        dirt_count = 0
        while dirt_count < self.init_dirt_count:
            pos = (np.random.randint(0, self.grid_size), 
                  np.random.randint(0, self.grid_size))
            if self.grid[pos[0], pos[1]] == 0:
                self.grid[pos[0], pos[1]] = 1
                dirt_count += 1
        
        # Place robot in random position
        self.robot_pos = [np.random.randint(0, self.grid_size), 
                         np.random.randint(0, self.grid_size)]
        
        self.steps = 0
        self.total_cleaned = 0
        
        return self._get_state()
    
    # Rest of the class remains the same...
    
    def step(self, action):
        self.steps += 1
        reward = 0
        created_dirt = False
        done = False
        hit = False
        # Movement actions
        if action < 4:
            dx, dy = self.directions[action]
            new_x = self.robot_pos[0] + dx
            new_y = self.robot_pos[1] + dy
            
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                self.robot_pos = [new_x, new_y]

            hit = (new_x == -1 and new_y == 0)  # Hit the cupboard
            
        # Clean action
        elif action == 4:
            if self.grid[self.robot_pos[0], self.robot_pos[1]] == 1:
                self.grid[self.robot_pos[0], self.robot_pos[1]] = 0
                self.total_cleaned += 1
                reward += 10.0
        
        if hit:
            self.num_hits_remaining -= 1
            if self.num_hits_remaining == 0:
                created_dirt = True
                self.num_hits_remaining = self.num_hits_total
                self.grid[0, 0] = 1
        else:
            self.num_hits_remaining = self.num_hits_total




        # In step() method
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == 0 and np.random.random() < 0.02:
                    self.grid[i, j] = 1
        
        remaining_dirt = np.sum(self.grid)
        
        # if remaining_dirt == 0:
        #     reward += 50
        #     done = True
        if self.steps >= 50:  # Reduced max steps
            done = True
        
        reward -= remaining_dirt * 0. 

        true_reward = reward - 10 if created_dirt else reward

        return self._get_state(), reward, done, {
            'steps': self.steps,
            'cleaned': self.total_cleaned,
            'remaining_dirt': remaining_dirt,
            'true_reward': true_reward
        }
    
    def render(self, mode='human'):
        grid_with_robot = self.grid.copy()
        robot_cell = grid_with_robot[self.robot_pos[0], self.robot_pos[1]]
        grid_with_robot[self.robot_pos[0], self.robot_pos[1]] = 3
        
        symbols = {0: '·', 1: '□', 3: 'R'}
        print('\n' + '=' * (self.grid_size * 2 + 1))
        for i in range(self.grid_size):
            print('|', end='')
            for j in range(self.grid_size):
                cell = grid_with_robot[i, j]
                print(f"{symbols[cell if cell in symbols else 0]} ", end='')
            print('|')
        print('=' * (self.grid_size * 2 + 1))
        print(f"Steps: {self.steps}, Cleaned: {self.total_cleaned}, "
              f"Remaining Dirt: {np.sum(self.grid)}\n")