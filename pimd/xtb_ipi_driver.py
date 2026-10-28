# xtb_ipi_driver.py

import sys
import socket
import numpy as np
from ase import units
from xtb.ase.calculator import XTB
from ase import Atoms

# --- Configuration ---
HOST = 'localhost'
PORT = 12345  # Must match the port in i-PI's input.xml
USE_UNIX_SOCKET = False  # Set to True if using a Unix socket

# --- Setup the xTB Calculator ---
calc = XTB(method="GFN0-xTB", accuracy=1.0, electronic_temperature=300.0)

# --- Helper to convert a string of numbers to a numpy array ---
def parse_coords(lines):
    """Parses coordinates from the i-PI POSDATA block."""
    # The first line is the cell (we'll ignore for now)
    # The second line is the number of atoms
    # The rest are the coordinates
    return np.array([[float(x) for x in line.split()] for line in lines])

# --- Main Driver Loop ---
def run_driver():
    # 1. Create a socket and connect to i-PI
    sock = socket.socket(socket.AF_UNIX if USE_UNIX_SOCKET else socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    sock_file = sock.makefile('r')
    print(f"Connected to i-PI server at {HOST}:{PORT}")

    # 2. Tell i-PI we're ready
    print("READY")
    sock.sendall(b"READY\n")

    # 3. Main loop: listen for commands
    while True:
        line = sock_file.readline()
        if not line:
            break
        tokens = line.strip().split()
        if not tokens:
            continue

        command = tokens[0]

        if command == 'STATUS':
            # i-PI is asking for our status. We're ready.
            sock.sendall(b"READY\n")
        elif command == 'GETFORCE':
            # i-PI is sending new positions and wants forces back.
            # We need to read the POSDATA block.
            # A POSDATA block is structured as:
            #   POSDATA
            #   <cell_1> <cell_2> ... <cell_9>
            #   <number_of_atoms>
            #   <atom1_x> <atom1_y> <atom1_z>
            #   ...
            #   END
            pos_lines = []
            while True:
                pos_line = sock_file.readline().strip()
                if pos_line == "END":
                    break
                pos_lines.append(pos_line)

            # Parse the coordinates and create an ASE Atoms object
            # (For simplicity, we assume the cell and number of atoms are
            #  correctly formatted. A production driver should parse them.)
            try:
                # We need the cell and the number of atoms.
                # The first line is the cell, second is natoms, rest are positions.
                # This is a simplified example.
                # A real implementation would use a robust parser.
                cell_strings = pos_lines[0].split()
                cell = np.array([float(x) for x in cell_strings]).reshape(3,3)
                natoms = int(pos_lines[1])
                positions = []
                for line in pos_lines[2:2+natoms]:
                    positions.append([float(x) for x in line.split()])
                atoms = Atoms(symbols='H2O'* (natoms//3), positions=positions, cell=cell, pbc=True)
                atoms.calc = calc

                # Compute forces
                forces = atoms.get_forces()  # This calls xTB

                # Send forces back to i-PI
                sock.sendall(b"FORCEREADY\n")
                sock.sendall(f"{forces.shape[0]}\n".encode())
                for force in forces:
                    sock.sendall(f"{force[0]:20.10f} {force[1]:20.10f} {force[2]:20.10f}\n".encode())
                sock.sendall(b"END FORCE\n")
            except Exception as e:
                # If something goes wrong, tell i-PI we had an error
                sock.sendall(b"ERROR\n")
                print(f"Error during force calculation: {e}", file=sys.stderr)
        elif command == 'EXIT':
            # i-PI is shutting down. Close the connection and exit.
            print("Received EXIT command. Shutting down.")
            break

    sock.close()
    print("Driver finished.")

if __name__ == "__main__":
    run_driver()