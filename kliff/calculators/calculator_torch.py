from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from loguru import logger
from torch.utils.data import DataLoader

from kliff.dataset.dataset import Configuration
from kliff.dataset.dataset_torch import FingerprintsDataset, fingerprints_collate_fn
from kliff.models.model_torch import ModelTorch
from kliff.models.neural_network import NeuralNetwork
from kliff.utils import pickle_load, to_path


class CalculatorTorch:
    """
    A calculator for torch based models.

    Args:
        model: torch models, e.g. :class:`~kliff.neuralnetwork.NeuralNetwork`.
        gpu: whether to use gpu for training. If `int` (e.g. 0), will trained on this
            gpu device. If `True` will always train on gpu `0`.
    """

    implemented_property = ["energy", "forces", "stress"]

    def __init__(self, model: ModelTorch, gpu: Union[bool, int] = None):
        device = _get_device(gpu)
        self._model = model.to(device)

        self.dtype = self.model.descriptor.dtype
        self.fingerprints_path = None

        self.use_energy = None
        self.use_forces = None
        self.use_stress = None

        self.results = dict([(i, None) for i in self.implemented_property])

    def create(
        self,
        configs: List[Configuration],
        use_energy: bool = True,
        use_forces: bool = True,
        use_stress: bool = False,
        fingerprints_filename: Union[Path, str] = "fingerprints.pkl",
        fingerprints_mean_stdev_filename: Optional[Union[Path, str]] = None,
        reuse: bool = False,
        use_welford_method: bool = False,
        nprocs: int = 1,
    ):
        """
        Process configs to generate fingerprints.

        Args:
            configs: atomic configurations
            use_energy: Whether to require the calculator to compute energy.
            use_forces: Whether to require the calculator to compute forces.
            use_stress: Whether to require the calculator to compute stress.
            fingerprints_filename: Path to save the generated fingerprints.
                If `reuse=True`, Will not generate the fingerprints, but directly use the
                one provided via this file.
            fingerprints_mean_stdev_filename: Path to save the mean and standard deviation
                of the fingerprints. If `reuse=True`, Will not generate new fingerprints
                mean and stdev, but directly use the one provided via this file.
                If `normalize` is not required by a descriptor, this is ignored.
            reuse: Whether to reuse provided fingerprints.
            use_welford_method: Whether to compute mean and standard deviation using the
                Welford method, which is memory efficient. See
                https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            nprocs: Number of processes used to generate the fingerprints. If `1`, run
                in serial mode, otherwise `nprocs` processes will be forked via
                multiprocessing to do the work.
        """

        self.configs = configs
        self.use_energy = use_energy
        self.use_forces = use_forces
        self.use_stress = use_stress

        if isinstance(configs, Configuration):
            configs = [configs]

        # reuse existing file
        if reuse:
            self.fingerprints_path = to_path(fingerprints_filename)
            if not self.fingerprints_path.exists():
                raise CalculatorTorchError(
                    f"You specified `reuse=True` to reuse the fingerprints stored in "
                    f"`{self.fingerprints_path}` This file does not exists."
                )
            logger.info(f"Reuse fingerprints `{self.fingerprints_path}`")

            if self.model.descriptor.normalize:
                path = (
                    None
                    if fingerprints_mean_stdev_filename is None
                    else to_path(fingerprints_mean_stdev_filename)
                )

                if path is None or not path.exists():
                    raise CalculatorTorchError(
                        f"You specified `reuse=True` to reuse the fingerprints. The "
                        f"mean and stdev file of the fingerprints `{path}` does not "
                        "exists."
                    )

                self.model.descriptor.load_state_dict(pickle_load(path))

                logger.info(f"Reuse fingerprints mean and stdev `{path}`")

        # generate fingerprints and pickle it
        else:
            self.fingerprints_path = self.model.descriptor.generate_fingerprints(
                configs,
                use_forces,
                use_stress,
                fingerprints_filename,
                fingerprints_mean_stdev_filename,
                use_welford_method,
                nprocs,
            )

    def get_compute_arguments(self, batch_size: int = 1):
        """
        Return the dataloader with batch size set to `batch_size`.
        """
        fname = self.fingerprints_path
        fp = FingerprintsDataset(fname)
        loader = DataLoader(
            dataset=fp, batch_size=batch_size, collate_fn=fingerprints_collate_fn
        )

        return loader

    def fit(self):
        path = self.fingerprints_path
        self.model.fit(path)

    def compute(self, batch):

        #
        # shape N--number of atoms in a config; D--feature dim
        # zeta: (N, D)
        # dzetadr_force: (N, D, 3N)
        # dzetadr_stress: (N, D, 6)
        #
        # batching dzetadr_force seems difficult, because two axes have different size
        # this seems doable, combine N and 3N as one dim, and use einstein sum

        device = self.model.device

        grad = self.use_forces or self.use_stress

        # TODO, the batching should be moved to dataloader
        # get information from batch
        zeta_config = [sample["zeta"] for sample in batch]
        zeta_stacked = torch.cat(zeta_config, dim=0).to(device)

        # evaluate model
        if grad:
            zeta_stacked.requires_grad_(True)

        energy_atom = self.model(zeta_stacked)

        # forces and stress
        if not self.use_forces:
            forces_config = None
        else:
            forces_config = []
        if not self.use_stress:
            stress_config = None
        else:
            stress_config = []

        natoms_config = [len(zeta) for zeta in zeta_config]
        energy_config = [e.sum() for e in torch.split(energy_atom, natoms_config)]

        if grad:
            dedzeta = torch.autograd.grad(
                energy_atom.sum(), zeta_stacked, create_graph=True
            )[0]
            zeta_stacked.requires_grad_(False)  # no need of grad any more

            dedzeta_config = torch.split(dedzeta, natoms_config)

            for i, sample in enumerate(batch):
                dedz = dedzeta_config[i]

                if self.use_forces:
                    dzetadr_forces = sample["dzetadr_forces"].to(device)
                    f = self._compute_forces(dedz, dzetadr_forces)
                    forces_config.append(f)

                if self.use_stress:
                    dzetadr_stress = sample["dzetadr_stress"].to(device)
                    volume = sample["dzetadr_volume"]
                    s = self._compute_stress(dedz, dzetadr_stress, volume)
                    stress_config.append(s)

        self.results["energy"] = energy_config
        self.results["forces"] = forces_config
        self.results["stress"] = stress_config
        return {
            "energy": energy_config,
            "forces": forces_config,
            "stress": stress_config,
        }

    @property
    def model(self):
        """Get the underlying torch model"""
        return self._model

    def save_model(self, epoch: int, force_save: bool = False):
        """
        Save the model to disk.

        When to save a model is dependent on `epoch` and a model's metadata for save.

        Args:
            epoch: current optimization epoch.
            force_save: save the model, ignoring `epoch` and save metadata.
        """
        # save metadata
        save_prefix = self.model.save_prefix
        save_start = self.model.save_start
        save_frequency = self.model.save_frequency

        path = to_path(save_prefix).joinpath(f"model_epoch{epoch}.pkl")
        if force_save:
            self.model.save(path)
        else:
            if epoch >= save_start and (epoch - save_start) % save_frequency == 0:
                self.model.save(path)

    def get_energy(self, batch):
        return self.results["energy"]

    def get_forces(self, batch):
        return self.results["forces"]

    def get_stress(self, batch):
        return self.results["stress"]

    @staticmethod
    def _compute_forces(denergy_dzeta, dzetadr):
        forces = -torch.tensordot(denergy_dzeta, dzetadr, dims=([0, 1], [0, 1]))
        return forces

    @staticmethod
    def _compute_stress(denergy_dzeta, dzetadr, volume):
        forces = torch.tensordot(denergy_dzeta, dzetadr, dims=([0, 1], [0, 1])) / volume
        return forces


class CalculatorTorchSeparateSpecies(CalculatorTorch):
    """
    A calculator supporting models of difference species.

    Args:
        models: {species: model} with species specifying the chemical symbol for the
            model.
        gpu: whether to use gpu for training. If `int` (e.g. 0), will trained on this
            gpu device. If `True` will always train on gpu `0`.
    """

    def __init__(self, models: Dict[str, NeuralNetwork], gpu: Union[bool, int] = None):
        device = _get_device(gpu)

        self.models = models

        self.dtype = None
        for s, m in self.models.items():
            m.to(device)

            if self.dtype is None:
                self.dtype = m.descriptor.dtype
            else:
                if self.dtype != m.descriptor.dtype:
                    raise CalculatorTorchError("inconsistent `dtype` from descriptors.")

        self._model = _ModelWrapper(models)

        self.fingerprints_path = None

        self.use_energy = None
        self.use_forces = None
        self.use_stress = None

        self.results = dict([(i, None) for i in self.implemented_property])

    def compute(self, batch):

        device = self.model.device

        grad = self.use_forces or self.use_stress

        # collate batch by species

        supported_species = self.models.keys()
        zeta_by_species = {s: [] for s in supported_species}
        config_id_by_species = {s: [] for s in supported_species}
        zeta_config = []

        for i, sample in enumerate(batch):
            zeta = sample["zeta"].to(device)
            species = sample["configuration"].species
            zeta.requires_grad_(True)
            zeta_config.append(zeta)

            for s, z in zip(species, zeta):
                if s not in supported_species:
                    raise CalculatorTorchError(f"No model for species: {s}")
                else:
                    zeta_by_species[s].append(z)
                    config_id_by_species[s].append(i)

        # evaluate model to compute energy
        energy_config = [None for _ in range(len(batch))]
        for s, zeta in zeta_by_species.items():

            # have no species "s" in this batch of data
            if not zeta:  # zeta == []
                continue

            z_tensor = torch.stack(zeta)  # convert a list of tensor to tensor
            energy = self.models[s](z_tensor)

            for e_atom, i in zip(energy, config_id_by_species[s]):
                if energy_config[i] is None:
                    energy_config[i] = e_atom
                else:
                    # note cannot use +=, energy e_atom is a view
                    energy_config[i] = energy_config[i] + e_atom

        # forces and stress
        if not self.use_forces:
            forces_config = None
        else:
            forces_config = []
        if not self.use_stress:
            stress_config = None
        else:
            stress_config = []
        if grad:
            for i, sample in enumerate(batch):

                # derivative of energy w.r.t. zeta
                energy = energy_config[i]
                zeta = zeta_config[i]
                dedz = torch.autograd.grad(energy, zeta, create_graph=True)[0]
                zeta.requires_grad_(False)  # no need of grad any more

                if self.use_forces:
                    dzetadr_forces = sample["dzetadr_forces"].to(device)
                    f = self._compute_forces(dedz, dzetadr_forces)
                    forces_config.append(f)

                if self.use_stress:
                    dzetadr_stress = sample["dzetadr_stress"]
                    volume = sample["dzetadr_volume"].to(device)
                    s = self._compute_stress(dedz, dzetadr_stress, volume)
                    stress_config.append(s)

        self.results["energy"] = energy_config
        self.results["forces"] = forces_config
        self.results["stress"] = stress_config
        return {
            "energy": energy_config,
            "forces": forces_config,
            "stress": stress_config,
        }

    @property
    def model(self):
        return self._model

    def save_model(self, epoch: int, force_save: bool = False):
        """
        Save the models to disk.

        When to save a model is dependent on `epoch` and a model's metadata for save.

        Args:
            epoch: current optimization epoch.
            force_save: save the model, ignoring `epoch` and save metadata.
        """
        # save metadata
        for name, model in self.models.items():
            save_prefix = model.save_prefix
            save_start = model.save_start
            save_frequency = model.save_frequency

            path = to_path(save_prefix).joinpath(f"model_{name}_epoch{epoch}.pkl")
            if force_save:
                model.save(path)
            else:
                if epoch >= save_start and (epoch - save_start) % save_frequency == 0:
                    model.save(path)


class _ModelWrapper(torch.nn.Module):
    """
    A wrapper over multiple torch models.

    Only add necessary properties:
      - `LossNeuralNetworkModel` uses `calculator.model.parameters()` and
      - `calculator.model.device`, and the model wrapper only need to provide them.
      - descriptor: needed by model create
    """

    def __init__(self, models: Dict[str, torch.nn.Module]):
        super().__init__()
        self._models = torch.nn.ModuleDict(models)

        first_model = list(models.values())[0]

        # Assuming all models using the same descriptor as in the example_NN_SiC.py
        # example, then it's OK to set it to the descriptor of the first model.
        self._descriptor = first_model.descriptor

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def descriptor(self):
        return self._descriptor


# class CalculatorTorchDDP(CalculatorTorch):
#     def __init__(self, model, rank, world_size):
#         super(self).__init__(model)
#         self.set_up(rank, world_size)
#
#     def set_up(self, rank, world_size):
#         os.environ["MASTER_ADDR"] = "localhost"
#         os.environ["MASTER_PORT"] = "12355"
#         dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
#     def clean_up(self):
#         dist.destroy_process_group()
#
#     def compute(self, batch):
#         grad = self.use_forces
#
#         # collate batch input to NN
#         zeta_config = self._collate(batch, "zeta")
#         if grad:
#             for zeta in zeta_config:
#                 zeta.requires_grad_(True)
#         zeta_stacked = torch.cat(zeta_config, dim=0)
#
#         # evaluate model
#         model = DistributedDataParallel(self.model)
#         energy_atom = model(zeta_stacked)
#
#         # energy
#         natoms_config = [len(zeta) for zeta in zeta_config]
#         energy_config = [e.sum() for e in torch.split(energy_atom, natoms_config)]
#
#         # forces
#         if grad:
#             dzetadr_config = self._collate(batch, "dzetadr")
#             forces_config = self.compute_forces_config(
#                 energy_config, zeta_config, dzetadr_config
#             )
#             for zeta in zeta_config:
#                 zeta.requires_grad_(False)
#         else:
#             forces_config = None
#
#         return {"energy": energy_config, "forces": forces_config}
#
#     def __del__(self):
#         self.clean_up()


class CalculatorTorchError(Exception):
    def __init__(self, msg):
        super(CalculatorTorchError, self).__init__(msg)
        self.msg = msg


def _get_device(gpu):

    device = None
    if isinstance(gpu, bool):
        if gpu:
            device = torch.device(0)
            logger.info(f"Training on gpu")
    elif isinstance(gpu, int):
        device = torch.device(gpu)
        logger.info(f"Training on gpu {gpu}")
    if device is None:
        logger.info("Training on cpu")

    return device
