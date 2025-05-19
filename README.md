## Installation

```bash
git clone https://github.com/TTomilin/DIC.git
cd DIC
```

### Conda (recommended)
```bash
conda create -n dqn_demo python=3.11
conda activate dqn_demo
conda install conda-forge::gcc
```

### Run env
```bash
pip install .
```

### Train
```bash
pip install .[train]
```

## Run

```
python scripts/1_run_gym_cartpole.py
```

                 |       :     . |  
                 | '  :      '   |
                 |  .  |   '  |  |
       .--._ _...:.._ _.--. ,  ' |
      (  ,  `        `  ,  )   . |
       '-/              \-'  |   |
         |  o   /\   o  |       :|
         \     _\/_     / :  '   |
         /'._   ^^   _.;___      |
       /`    `""""""`      `\=   |
     /`                     /=  .|
    ;             '--,-----'=    |
    |                 `\  |    . |
    \                   \___ :   |
    /'.                     `\=  |
    \_/`--......_            /=  |
                |`-.        /= : |
                | : `-.__ /` .   |
                |    .   ` |    '|
                |  .  : `   . |  |

