import os
import multiprocessing as mp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallelCPU
import kliff
from ..dataset.dataset import Configuration

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
        if use_stress:
            raise NotImplementedError('"stress" is not supported by NN calculator.')

        self.configs = configs
        self.use_energy = use_energy
        self.use_forces = use_forces

        if isinstance(configs, Configuration):
            configs = [configs]

        # generate pickled fingerprints
        print('Start generating fingerprints')
        fname = self.model.descriptor.generate_train_fingerprints(
            configs, grad=use_forces, reuse=reuse, nprocs=nprocs
        )
        print('Finish generating fingerprints')
        self.train_fingerprints_path = fname

    def get_train_fingerprints_path(self):
        """Return the path to the training set fingerprints: `train.pkl`."""
        return self.train_fingerprints_path

    def fit(self):
        path = self.get_train_fingerprints_path()
        self.model.fit(path)

    def compute(self, batch):
        grad = self.use_forces

        # collate batch input to NN
        zeta_config = self._collate(batch, 'zeta')
        if grad:
            for zeta in zeta_config:
                zeta.requires_grad_(True)
        zeta_batch = torch.cat(zeta_config, dim=0)

        # evaluate model
        energy_atom = self.model(zeta_batch)

        # energy
        natoms_config = [len(zeta) for zeta in zeta_config]
        energy_config = [e.sum() for e in torch.split(energy_atom, natoms_config)]

        # forces
        if grad:
            dzeta_dr_config = self._collate(batch, 'dzeta_dr')
            forces_config = self.get_forces_config(
                energy_config, zeta_config, dzeta_dr_config
            )
            for zeta in zeta_config:
                zeta.requires_grad_(False)
            # zeta_batch.requires_grad_(False)
        else:
            forces_config = None

        return {'energy': energy_config, 'forces': forces_config}

    @staticmethod
    def _collate(batch, key):
        return [sample[key] for sample in batch]

    @staticmethod
    def get_forces_config(energy_config, zeta_config, dzeta_dr_config):
        def compute_forces(energy, zeta, dzeta_dr):
            denergy_dzeta = torch.autograd.grad(energy, zeta, create_graph=True)[0]
            forces = -torch.tensordot(denergy_dzeta, dzeta_dr, dims=([0, 1], [0, 1]))
            return forces

        forces = []
        for energy, zeta, dzeta_dr in zip(energy_config, zeta_config, dzeta_dr_config):
            f = compute_forces(energy, zeta, dzeta_dr)
            forces.append(f)
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
        zeta_batch = torch.cat(zeta_config, dim=0)

        # evaluate model
        model = DistributedDataParallelCPU(self.model)
        energy_atom = model(zeta_batch)

        # energy
        natoms_config = [len(zeta) for zeta in zeta_config]
        energy_config = [e.sum() for e in torch.split(energy_atom, natoms_config)]

        # forces
        if grad:
            dzeta_dr_config = self._collate(batch, 'dzeta_dr')
            forces_config = self.get_forces_config(
                energy_config, zeta_config, dzeta_dr_config
            )
            for zeta in zeta_config:
                zeta.requires_grad_(False)
            # zeta_batch.requires_grad_(False)
        else:
            forces_config = None

        return {'energy': energy_config, 'forces': forces_config}

    def __del__(self):
        self.clean_up()
