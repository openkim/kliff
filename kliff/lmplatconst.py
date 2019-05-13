import subprocess
import os


def lmp_lat_const(modelname):
    lmp_input_str = """# Define unit set and class of atomic model
  units metal
  atom_style atomic

  # Periodic boundary conditions along all three dimensions
  boundary p p p

  # lattice
  lattice   custom  3.2 &
            a1 1.0 0.0 0.0   a2 0.5 $(sqrt(3.0)/2.0) 0.0  a3 0.0 0.0 4.0 &
            basis 0.0 0.0 0.5 basis $(1.0/3.0) $(1.0/3.0) 0.375  basis $(1.0/3.0) $(1.0/3.0) 0.625 &
            spacing 1.0 $(sqrt(3.0)/2.0) 1.0  # lattice spacing is relative to lattice constant

  #this will create a tilted simbox
  region       simbox prism 0 1 0 1 0 4 0.5 0 0  units lattice  # (unit relative to lattice const )

  # create simbox that allows 2 types of atoms
  create_box     2 simbox

  # create atoms such that basis 1 is of type 1, and basis 2 and 3 are of type 2
  # the type (1) before box is not working here
  create_atoms   1 box basis 1 1 basis 2 2 basis 3 2

  mass      1 95.94
  mass      2 32.065

  # Specify which KIM Model to use, letting LAMMPS compute the virial/pressure
  pair_style    kim LAMMPSvirial rpls_modelname
  pair_coeff    * * Mo S

  # set pressure in x and y direction, so as to relax boxsize
  fix       1 all box/relax x 0.0 y 0.0

  # Set what information to write to dump file
  #dump      id all custom 1 lammps.dump id type x y z fx fy fz
  #dump_modify   id every 1 format "%d %d %13.5e %13.5e %13.5e %13.5e %13.5e %13.5e"

  # minimize energy
  minimize  1.0e-15 1.0e-15 10000 10000

  variable    mylx equal lx
  print "lat_const = ${mylx}"
  """

    # create lammps input file
    lmp_input_str = lmp_input_str.replace('rpls_modelname', modelname)
    with open('lammps.in', 'w') as fout:
        fout.write(lmp_input_str)

    # run lammps
    subprocess.call('lmp_serial <lammps.in > lammps.out', shell=True)

    # write results to edn format
    with open('lammps.out', 'r') as fin:
        for line in fin:
            if 'lat_const' in line:
                lat_const = float(line.split('=')[1])

    with open('lattice_const.edn', 'w') as fout:
        edn_str = '''{
      "species" {
        "source-value"  ["Mo" "Mo" "S"]
      }

      "lattice-const" {
        "source-unit" "Angstrom"
        "source-value"  %22.15e
      }
    }''' % (
            lat_const
        )

        fout.write(edn_str)


if __name__ == '__main__':
    lmp_lat_const()
