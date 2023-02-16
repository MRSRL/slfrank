import simple_parsing as sp
import dataclasses as dc
import pathlib as plib


@dc.dataclass
class Config(sp.helpers.Serializable):
    configFile: str = sp.field(default="", alias=["-c"])
    outputPath: str = sp.field(default="./pulses/", alias=["-o"])

    numSamples: int = sp.field(default=300, alias=["-n"])
    timeBandwidth: float = sp.field(default=2.75, alias=["-tb"])
    desiredDuration: int = sp.field(default=0, alias=["-dur"])              # us
    desiredSliceGrad: float = sp.field(default=35.0, alias=["-sg"])         # [mT/m]
    desiredSliceThickness: float = sp.field(default=0.7, alias=["-st"])     # [mm]
    rippleSizes: float = sp.field(default=0.01, alias=["-r"])
    pulseType: str = sp.choice("ex", "se", "inv", default="ex", alias=["-pu"])
    phaseType: str = sp.choice("linear", "minimum", default="linear", alias=["-ph"])
    maxIter: int = sp.field(2500, alias=["-mi"])

    @classmethod
    def from_cmd_args(cls, args: sp.ArgumentParser.parse_args):
        # create default_dict
        default_instance = cls()
        instance = cls()
        if args.config.ConfigFile:
            confPath = plib.Path(args.config.ConfigFile).absolute()
            instance = cls.load(confPath)
            # might contain defaults
        for key, item in default_instance.__dict__.items():
            parsed_arg = args.config.__dict__.get(key)
            # if parsed arguments are not defaults
            if parsed_arg != default_instance.__dict__.get(key):
                # update instance even if changed by config file -> that way prioritize cmd line input
                instance.__setattr__(key, parsed_arg)
        return instance


def createCommandlineParser():
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = sp.ArgumentParser(prog='slr_slfrank')
    parser.add_arguments(Config, dest="config")
    args = parser.parse_args()

    return parser, args


if __name__ == '__main__':
    save_path = plib.Path("./js/default_conf.json").absolute()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    conf = Config()
    conf.save(save_path)

