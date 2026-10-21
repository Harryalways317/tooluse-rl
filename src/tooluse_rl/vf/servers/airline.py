from tooluse_rl.vf.toolset import make_toolset_class

Tau2AirlineToolset = make_toolset_class("airline", module=__name__)

if __name__ == "__main__":
    Tau2AirlineToolset.run()
