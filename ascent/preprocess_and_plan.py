import hydra
from omegaconf import DictConfig

from ascent import setup_root

setup_root()

from ascent import utils

log = utils.get_pylogger(__name__)


def preprocess_and_plan(cfg: DictConfig) -> None:
    """Preprocess dataset and plan experiments.

    Args:
        cfg: Configuration composed by Hydra.
    """

    log.info(f"Instantiating preprocessor <{cfg.preprocessor._target_}>")
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)

    log.info("Start preprocessing...")
    preprocessor.run()

    if cfg.pl2d:
        log.info(f'Instantiating 2D planner <{cfg.planner["planner2d"]._target_}>')
        planner2d = hydra.utils.instantiate(cfg.planner["planner2d"])
        planner2d.plan_experiment()

    if cfg.pl3d:
        log.info(f'Instantiating 3D planner <{cfg.planner["planner3d"]._target_}>')
        planner3d = hydra.utils.instantiate(cfg.planner["planner3d"])
        planner3d.plan_experiment()


@hydra.main(version_base="1.3", config_path="configs", config_name="preprocess_and_plan")
def main(cfg: DictConfig) -> None:
    preprocess_and_plan(cfg)


if __name__ == "__main__":
    main()
