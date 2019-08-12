import os
import multiprocessing as mp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallelCPU
from torch.utils.data import DataLoader
import kliff
from ..dataset.dataset import Configuration
from ..dataset.dataset_torch import FingerprintsDataset, fingerprints_collate_fn


logger = kliff.logger.get_logger(__name__)


class CalculatorTorch:
    """ A neural network calculator.

    Parameters
    ----------
    model: obj
        Instance of :class:`~kliff.neuralnetwork.NeuralNetwork`.

    Attributes
    ----------
    attr1: list
        This is an example attribute.
    """

    # TODO should be moved to Model
    implemented_property = ['energy', 'forces']

    def __init__(self, model):

        self.model = model
        self.dtype = self.model.descriptor.dtype
        self.train_fingerprints_path = None

        self.use_energy = None
        self.use_forces = None
        self.use_stress = None

        self.results = dict([(i, None) for i in self.implemented_property])

    def create(
        self,
        configs,
        use_energy=True,
        use_forces=True,
        use_stress=False,
        reuse=False,
        nprocs=mp.cpu_count(),
    ):
        """Process configs into fingerprints.

        Parameters
        ----------

        configs: list of Configuration object

        use_energy: bool (optional)
            Whether to require the calculator to compute energy.

        use_forces: bool (optional)
            Whether to require the calculator to compute forces.

        use_stress: bool (optional)
            Whether to require the calculator to compute stress.

        nprocs: int (optional)
            Number if processors.

        """

        self.configs = configs
        self.use_energy = use_energy
        self.use_forces = use_forces
        self.use_stress = use_stress

        if isinstance(configs, Configuration):
            configs = [configs]

        # generate pickled fingerprints
        fname = self.model.descriptor.generate_train_fingerprints(
            configs,
            fit_forces=use_forces,
            fit_stress=use_stress,
            reuse=reuse,
            nprocs=nprocs,
        )
        self.train_fingerprints_path = fname

    def get_train_fingerprints_path(self):
        """Return the path to the training set fingerprints: `train.pkl`."""
        return self.train_fingerprints_path

    def get_compute_arguments(self, batch_size=1):
        """Return a list of compute arguments, each associated with a configuration.
        """

        fname = self.get_train_fingerprints_path()
        fp = FingerprintsDataset(fname)
        loader = DataLoader(
            dataset=fp, batch_size=batch_size, collate_fn=fingerprints_collate_fn
        )

        return loader

    def fit(self):
        path = self.get_train_fingerprints_path()
        self.model.fit(path)

    def compute(self, batch):

        grad = self.use_forces or self.use_stress

        # collate batch input to NN
        zeta_config = [sample['zeta'] for sample in batch]
        if grad:
            for zeta in zeta_config:
                zeta.requires_grad_(True)

        # evaluate model (forward pass)
        zeta_stacked = torch.cat(zeta_config, dim=0)
        energy_atom = self.model(zeta_stacked)

        # energy
        natoms_config = [len(zeta) for zeta in zeta_config]
        energy_config = [e.sum() for e in torch.split(energy_atom, natoms_config)]

        # forces and stress (backward propagation)
        forces_config = []
        stress_config = []
        if grad:
            for i, sample in enumerate(batch):

                # derivative of energy w.r.t. zeta
                energy = energy_config[i]
                zeta = zeta_config[i]
                dedz = torch.autograd.grad(energy, zeta, create_graph=True)[0]
                zeta.requires_grad_(False)  # no need of grad any more

                if self.use_forces:
                    dzetadr_forces = sample['dzetadr_forces']
                    f = self.compute_forces(dedz, dzetadr_forces)
                    forces_config.append(f)

                if self.use_stress:
                    dzetadr_stress = sample['dzetadr_stress']
                    volume = sample['dzetadr_volume']
                    s = self.compute_stress(dedz, dzetadr_stress, volume)
                    stress_config.append(s)

        if not self.use_forces:
            forces_config = None
        if not self.use_stress:
            stress_config = None

        return {'energy': energy_config, 'forces': forces_config, 'stress': stress_config}

    @staticmethod
    def compute_forces(denergy_dzeta, dzetadr):
        forces = -torch.tensordot(denergy_dzeta, dzetadr, dims=([0, 1], [0, 1]))
        return forces

    @staticmethod
    def compute_stress(denergy_dzeta, dzetadr, volume):
        forces = torch.tensordot(denergy_dzeta, dzetadr, dims=([0, 1], [0, 1])) / volume
        return forces


class CalculatorTorchDDPCPU(CalculatorTorch):
    def __init__(self, model, rank, world_size):
        super(CalculatorTorchDDPCPU, self).__init__(model)
        self.set_up(rank, world_size)

    def set_up(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('gloo', rank=rank, world_size=world_size)

    def clean_up(self):
        dist.destroy_process_group()

    def compute(self, batch):
        grad = self.use_forces

        # collate batch input to NN
        zeta_config = self._collate(batch, 'zeta')
        if grad:
            for zeta in zeta_config:
                zeta.requires_grad_(True)
        zeta_stacked = torch.cat(zeta_config, dim=0)

        # evaluate model
        model = DistributedDataParallelCPU(self.model)
        energy_atom = model(zeta_stacked)

        # energy
        natoms_config = [len(zeta) for zeta in zeta_config]
        energy_config = [e.sum() for e in torch.split(energy_atom, natoms_config)]

        # forces
        if grad:
            dzetadr_config = self._collate(batch, 'dzetadr')
            forces_config = self.compute_forces_config(
                energy_config, zeta_config, dzetadr_config
            )
            for zeta in zeta_config:
                zeta.requires_grad_(False)
            # zeta_stacked.requires_grad_(False)
        else:
            forces_config = None

        return {'energy': energy_config, 'forces': forces_config}

    def __del__(self):
        self.clean_up()
