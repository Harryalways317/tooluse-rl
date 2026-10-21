from tooluse_rl.vf.toolset import make_toolset_class

Tau2RetailToolset = make_toolset_class("retail", module=__name__)

if __name__ == "__main__":
    Tau2RetailToolset.run()
