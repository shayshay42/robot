# Robot Project Setup

This project uses a lightweight virtual environment that inherits PyTorch from your base conda environment to avoid duplicating large GPU libraries.

## Environment Setup

### Python Environment
- **Base**: Python 3.12.3 (from base conda)
- **Virtual Environment**: `robot-venv` with `--system-site-packages`
- **PyTorch**: 2.5.1 with CUDA 12.4 support (inherited from base conda)
- **GPU**: NVIDIA GeForce RTX 3070 ✅

### Installed Packages
- ✅ **MuJoCo** 3.3.5 - Physics simulation engine
- ✅ **Gymnasium** 1.2.0 - RL environment interface
- ✅ **Stable Baselines3** 2.7.0 - RL algorithms
- ✅ **OpenCV** 4.12.0 - Computer vision
- ✅ **RoboSuite** 1.5.1 - Robot manipulation environments (fully working!)

## How to Use

### Option 1: Run with VS Code Task
1. Open any Python file
2. Press `Ctrl+Shift+P`
3. Type "Run Task" and select "Run Python with Robot Environment"

### Option 2: Command Line
```powershell
C:\Users\shaya\robot\robot-venv\Scripts\python.exe your_script.py
```

### Option 3: Activate Virtual Environment
```powershell
.\robot-venv\Scripts\activate
python your_script.py
```

## Examples

- `test_robotics_env.py` - Test all package installations
- `example_gym_pytorch.py` - Simple PyTorch + Gymnasium example
- `example_robosuite.py` - RoboSuite + PyTorch robot manipulation demo

## Benefits of This Setup

- 🚀 **No PyTorch duplication** - Uses existing base conda installation
- 💾 **Space efficient** - Minimal additional storage needed
- 🔥 **GPU ready** - Full CUDA support inherited
- 📦 **Package isolation** - Robotics packages only in this project
- 🔄 **Easy to recreate** - Simple requirements.txt

## Troubleshooting

### NumPy Compatibility
The environment uses NumPy 1.26.4 for compatibility with base conda packages. This is intentional.

### RoboSuite Demos
You can run interactive demos with:
```powershell
C:\Users\shaya\robot\robot-venv\Scripts\python.exe -m robosuite.demos.demo_device_control --environment Lift --robots UR5e --device keyboard
```

Use arrow keys to control the robot, spacebar for gripper, Ctrl+Q to reset.

## VS Code Configuration

The project includes:
- `.vscode/settings.json` - Python interpreter configuration
- `.vscode/tasks.json` - Build and run tasks
- `requirements.txt` - Package dependencies

## Project Structure

```
robot/
├── .vscode/                    # VS Code configuration
│   ├── settings.json          # Python interpreter settings
│   └── tasks.json             # Build and run tasks
├── robot-venv/                 # Virtual environment with robotics packages
├── example_gym_pytorch.py      # PyTorch + Gymnasium demo
├── example_robosuite.py        # RoboSuite + PyTorch demo
├── test_robotics_env.py        # Environment verification script
├── requirements.txt            # Package dependencies
└── README.md                   # This file
```
