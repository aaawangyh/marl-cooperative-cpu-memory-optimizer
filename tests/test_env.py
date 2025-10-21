from src.env import MultiAgentCPUMemEnv
def test_env(): env=MultiAgentCPUMemEnv(); o=env.reset(); a=env.step([0]*4,[0]*2); assert 'cpu' in o
