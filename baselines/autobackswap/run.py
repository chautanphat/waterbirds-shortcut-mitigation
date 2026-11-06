import sys
from utils import Config


def updated_config(path, arg_dict):
    config = Config(path)
    # update config with command line arguments
    # if the attribute exists in config
    # else raise error
    for key, value in arg_dict.items():
        if hasattr(config, key):
            attr_type = type(getattr(config, key))
            if attr_type == bool:
                value = value.lower() == "true"
            elif attr_type == list:
                _attr_type = (
                    type(getattr(config, key)[0]) if getattr(config, key) else str
                )
                value = value.strip("[]").split(",")
                value = [_attr_type(v.strip()) for v in value]
            else:
                value = attr_type(value)
            setattr(config, key, value)
        else:
            raise KeyError(f"Config has no attribute '{key}'")
    return config


if __name__ == "__main__":

    # read from argument line
    arg_dict = {}
    for arg in sys.argv[1:]:
        key, value = arg.split("--")[1].split("=")
        arg_dict[key] = value

    if "mode" not in arg_dict:
        raise Exception("mode argument is missing")

    mode = arg_dict["mode"]
    del arg_dict["mode"]

    # Determine config file based on dataset_type if not using foundation models
    if mode not in ["clip", "sam3", "qwen3"]:
        if "dataset_type" in arg_dict:
            dataset_type = arg_dict["dataset_type"]
            dataset_type_l = dataset_type.lower()
            if "waterbird" in dataset_type_l:
                config_file = "config_wb.yaml"
            elif "spawrious" in dataset_type_l:
                config_file = "config_spawrious.yaml"
            elif "spurious_vehicles" in dataset_type_l:
                config_file = "config_spurious_vehicles.yaml"
            else:
                raise ValueError(f"unknown dataset_type: '{dataset_type}'")
        else:
            print(
                "No benchmark specified in commandline arguments, defaulting to Waterbirds."
            )
            config_file = "config_wb.yaml"

    # Baselines
    if mode == "erm":
        from methods.erm import erm

        config = updated_config(f"./methods/erm/{config_file}", arg_dict)
        erm.run(config)
    elif mode == "dfr":
        from methods.dfr import dfr

        config = updated_config(f"./methods/dfr/{config_file}", arg_dict)
        dfr.run(config)
    elif mode == "afr":
        from methods.afr import afr

        config = updated_config(f"./methods/afr/{config_file}", arg_dict)
        afr.run(config)
    elif mode == "group_dro":
        from methods.group_dro import group_dro

        config = updated_config(f"./methods/group_dro/{config_file}", arg_dict)
        group_dro.run(config)
    elif mode == "coral":
        from methods.coral import coral

        config = updated_config(f"./methods/coral/{config_file}", arg_dict)
        coral.run(config)
    # --- Chang et al. ---
    elif mode.startswith("chang"):
        config = updated_config(f"./methods/chang/{config_file}", arg_dict)
        mode = mode.split("_")
        if len(mode) == 1:
            from methods.chang import chang

            chang.run(config)
        elif mode[1] == "generate":
            from methods.chang import generate_cf_data

            generate_cf_data.run(config)
        else:
            raise ValueError(f"unknown mode: '{'_'.join(mode)}'")
    # --- Our method ---
    elif mode.startswith("ours"):
        config = updated_config(f"./methods/ours/{config_file}", arg_dict)
        mode = mode.split("_")
        if len(mode) == 1:
            from methods.ours import ours

            ours.run(config)
        elif mode[1] == "generate":
            from methods.ours import augment_data

            augment_data.run(config)
        elif mode[1] == "detector":
            from methods.ours.detector import train as train_detector

            train_detector.train(config)
        else:
            raise ValueError(f"unknown mode: '{'_'.join(mode)}'")
    # --- Foundation models ---
    elif mode == "sam3":
        from methods.sam3 import sam3

        config = updated_config("./methods/sam3/config.yaml", arg_dict)
        sam3.run(config)
    elif mode == "qwen3":
        from methods.qwen3 import qwen3

        config = updated_config("./methods/qwen3/config.yaml", arg_dict)
        qwen3.run(config)
    elif mode == "clip":
        from methods.clip import clip

        config = updated_config("./methods/clip/config.yaml", arg_dict)
        clip.run(config)

    else:
        raise ValueError(f"unknown mode: '{mode}'")
