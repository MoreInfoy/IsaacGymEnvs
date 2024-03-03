import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg['db']['driver'])
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
