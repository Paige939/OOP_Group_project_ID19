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

# Installation
```
pip install gymnasium[classic-control]
pip install pygame
```

# Introduction to Pendulum
