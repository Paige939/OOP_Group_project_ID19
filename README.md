# Project structure
```
project/
├── part1
├── part2
│   └── frozen_lake.py
└── part3
    ├── agents/
    │   ├── __init__.py
    │   ├── random_agent.py
    │   └── other_agent.py (you can add new agent.py here)
    ├── base_agent.py
    ├── manage.py
    └── main.py
```

_You can add any other _agent.py (ex: some algorithm implementation) into Agents folder that inherit act() and reset() functions in base_agent.py__

# 新增agent.py方法
寫完自己的Agent.py後:
1. 在Agent檔案夾的__init__.py中加入from .agent import agent_class_name 及 __all__=["RandomAgent","agent_class_name",...], 這樣的話在main.py中就可以直接在from Agents import RandomAgent, CEM_Agent後面加上自己的Agent class name
2. 在main.py中建造屬於自己Agent的環境
3. 測試時有三種render mode可以用: None, rgb_array, human(None=rgb_array, 因為gymnasium default mode是rgb_array)

# Installation
```
pip install gymnasium[classic-control]
pip install pygame
```

# Introduction to Pendulum
