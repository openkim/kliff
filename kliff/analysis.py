def error_report(self, normalize_by_num_atoms=True, fname='ERROR_REPORT'):
    """Write error of each configuration to fname.

        Parameters
        ----------
        fname: str
          path to the file to write the error report.
        """

    loss = self.get_loss(self.calculator.get_opt_params())

    with open(fname, 'w') as fout:
        fout.write('\n' + '=' * 80 + '\n')
        fout.write('Final loss: {:18.10e}\n\n'.format(loss))
        fout.write('=' * 80 + '\n')
        if normalize_by_num_atoms:
            fout.write(
                '(Loss, energy RMSE, and forces RMSE are normalized by '
                'number of atoms: Natoms.)\n\n'
            )
        fout.write(
            '      Loss       energy RMSE     forces RMSE  Natoms  '
            'config. identifier\n\n'
        )

        cas = self.calculator.get_compute_arguments()
        for ca in cas:

            # prediction data
            self.calculator.compute(ca)
            pred = self.calculator.get_prediction(ca)

            # reference data
            ref = self.calculator.get_reference(ca)

            conf = ca.conf
            identifier = conf.get_identifier()
            natoms = conf.get_number_of_atoms()

            compute_energy = ca.get_compute_flag('energy')
            compute_forces = ca.get_compute_flag('forces')
            if compute_energy:
                pred_energy = self.calculator.get_energy(ca)
                ref_energy = ca.conf.get_energy()
                energy_rmse = pred_energy - ref_energy
            else:
                energy_rmse = None
            if compute_forces:
                pred_forces = self.calculator.get_forces(ca)
                ref_forces = ca.conf.get_forces()
                forces_rmse = np.linalg.norm(pred_forces - ref_forces)
            else:
                forces_rmse = None

            residual = self.residual_fn(identifier, natoms, pred, ref, self.residual_data)
            loss = 0.5 * np.linalg.norm(residual) ** 2

            if normalize_by_num_atoms:
                nz = natoms
            else:
                nz = 1

            if energy_rmse is None:
                if forces_rmse is None:
                    fout.write(
                        '{:14.6e} {} {}   {}   {}\n'.format(
                            loss / nz,
                            energy_rmse / nz,
                            forces_rmse / nz,
                            natoms,
                            identifier,
                        )
                    )
                else:
                    fout.write(
                        '{:14.6e} {} {:14.6e}   {}   {}\n'.format(
                            loss / nz,
                            energy_rmse / nz,
                            forces_rmse / nz,
                            natoms,
                            identifier,
                        )
                    )
            else:
                if forces_rmse is None:
                    fout.write(
                        '{:14.6e} {:14.6e} {}   {}   {}\n'.format(
                            loss / nz,
                            energy_rmse / nz,
                            forces_rmse / nz,
                            natoms,
                            identifier,
                        )
                    )
                else:
                    fout.write(
                        '{:14.6e} {:14.6e} {:14.6e}   {}   {}\n'.format(
                            loss / nz,
                            energy_rmse / nz,
                            forces_rmse / nz,
                            natoms,
                            identifier,
                        )
                    )
